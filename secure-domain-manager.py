#!/usr/bin/env python3
import asyncio
import aiohttp
import aiodns
import boto3
import json
import ssl
import uuid
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

class SecureDomainManager:
    def __init__(self, domain: str, region: str = "us-east-2"):
        self.domain = domain
        self.region = region
        self.session = boto3.Session(region_name=region)
        self.secrets = self.session.client('secretsmanager')
        self.route53 = self.session.client('route53')
        self.acm = self.session.client('acm')
        self.ssl_context = self._create_ssl_context()
        self.resolver = aiodns.DNSResolver()
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger('SecureDomainManager')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(handler)
        return logger

    def _create_ssl_context(self) -> ssl.SSLContext:
        context = ssl.create_default_context()
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        return context

    async def _get_godaddy_credentials(self) -> Dict[str, str]:
        try:
            response = self.secrets.get_secret_value(
                SecretId=f'kaleidoscope/godaddy/production'
            )
            return json.loads(response['SecretString'])
        except Exception as e:
            self.logger.error(f"Failed to retrieve GoDaddy credentials: {e}")
            raise

    async def _create_https_session(self, credentials: Dict[str, str]) -> aiohttp.ClientSession:
        return aiohttp.ClientSession(
            base_url="https://api.godaddy.com/v1",
            headers={
                'Authorization': f'sso-key {credentials["key"]}:{credentials["secret"]}',
                'Content-Type': 'application/json'
            },
            connector=aiohttp.TCPConnector(ssl=self.ssl_context)
        )

    async def configure_domain(self, alb_dns: str):
        credentials = await self._get_godaddy_credentials()
        async with await self._create_https_session(credentials) as session:
            # Create Route53 hosted zone
            zone_response = self.route53.create_hosted_zone(
                Name=self.domain,
                CallerReference=str(uuid.uuid4()),
                HostedZoneConfig={
                    'Comment': f'Managed by SecureDomainManager - {datetime.now(timezone.utc)}'
                }
            )
            zone_id = zone_response['HostedZone']['Id']
            self.logger.info(f"Created Route53 hosted zone: {zone_id}")

            # Get AWS nameservers
            nameservers = zone_response['DelegationSet']['NameServers']

            # Update GoDaddy nameservers
            async with session.put(
                f'/domains/{self.domain}',
                json={'nameServers': nameservers},
                ssl=self.ssl_context
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self.logger.error(f"Failed to update nameservers: {error_text}")
                    raise Exception(f"Failed to update nameservers: {error_text}")

            # Create DNS records
            await self._create_dns_records(zone_id, alb_dns)
            await self._setup_ssl_certificate(session)
            await self._verify_dns_propagation(nameservers)

    async def _create_dns_records(self, zone_id: str, alb_dns: str):
        changes = {
            'Changes': [
                {
                    'Action': 'UPSERT',
                    'ResourceRecordSet': {
                        'Name': self.domain,
                        'Type': 'A',
                        'AliasTarget': {
                            'HostedZoneId': 'Z35SXDOTRQ7X7K',  # us-east-2 ALB zone ID
                            'DNSName': alb_dns,
                            'EvaluateTargetHealth': True
                        }
                    }
                },
                {
                    'Action': 'UPSERT',
                    'ResourceRecordSet': {
                        'Name': f'www.{self.domain}',
                        'Type': 'CNAME',
                        'TTL': 300,
                        'ResourceRecords': [{'Value': self.domain}]
                    }
                },
                {
                    'Action': 'UPSERT',
                    'ResourceRecordSet': {
                        'Name': f'*.{self.domain}',
                        'Type': 'CNAME',
                        'TTL': 300,
                        'ResourceRecords': [{'Value': self.domain}]
                    }
                }
            ]
        }

        try:
            self.route53.change_resource_record_sets(
                HostedZoneId=zone_id,
                ChangeBatch=changes
            )
            self.logger.info("Created Route53 DNS records")
        except Exception as e:
            self.logger.error(f"Failed to create DNS records: {e}")
            raise

    async def _setup_ssl_certificate(self, session: aiohttp.ClientSession):
        # Generate CSR and private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )

        csr = x509.CertificateSigningRequestBuilder().subject_name(x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, self.domain),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "KaleidoscopeAI")
        ])).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(self.domain),
                x509.DNSName(f"*.{self.domain}")
            ]),
            critical=False
        ).sign(private_key, hashes.SHA256())

        # Request certificate from ACM
        cert_arn = self.acm.request_certificate(
            DomainName=self.domain,
            ValidationMethod='DNS',
            SubjectAlternativeNames=[f"*.{self.domain}"],
            IdempotencyToken=str(uuid.uuid4())[:32]
        )['CertificateArn']

        self.logger.info(f"Requested ACM certificate: {cert_arn}")

        # Store private key in Secrets Manager
        self.secrets.create_secret(
            Name=f'kaleidoscope/ssl/{self.domain}/private-key',
            SecretString=private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ).decode('utf-8')
        )

    async def _verify_dns_propagation(self, nameservers: List[str]):
        max_attempts = 30
        attempt = 0
        while attempt < max_attempts:
            try:
                for ns in nameservers:
                    response = await self.resolver.query(self.domain, 'NS')
                    if ns in [str(rdata.host) for rdata in response]:
                        continue
                    raise Exception(f"Nameserver {ns} not found in DNS response")
                self.logger.info("DNS propagation verified")
                return
            except Exception as e:
                attempt += 1
                if attempt >= max_attempts:
                    self.logger.error(f"DNS propagation verification failed: {e}")
                    raise
                await asyncio.sleep(10)

async def main():
    domain_manager = SecureDomainManager("artificialthinker.com")
    await domain_manager.configure_domain("kaleidoscope-prod-alb.us-east-2.elb.amazonaws.com")

if __name__ == "__main__":
    asyncio.run(main())