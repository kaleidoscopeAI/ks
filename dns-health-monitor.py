async def monitor_dns_health(self):
    """Advanced DNS health monitoring system with automatic recovery"""
    health_metrics = {
        'response_times': [],
        'nameserver_status': {},
        'record_integrity': {},
        'propagation_status': {}
    }

    while True:
        try:
            # Check all nameservers
            nameservers = await self._get_route53_nameservers()
            for ns in nameservers:
                start_time = time.time()
                try:
                    response = await self.resolver.query(self.domain, 'NS')
                    response_time = time.time() - start_time
                    health_metrics['response_times'].append(response_time)
                    health_metrics['nameserver_status'][ns] = {
                        'status': 'healthy',
                        'response_time': response_time,
                        'last_check': datetime.datetime.now(datetime.timezone.utc)
                    }
                except Exception as e:
                    health_metrics['nameserver_status'][ns] = {
                        'status': 'error',
                        'error': str(e),
                        'last_check': datetime.datetime.now(datetime.timezone.utc)
                    }
                    await self._handle_nameserver_failure(ns)

            # Verify record integrity
            await self._verify_dns_records(health_metrics)
            
            # Check propagation across public DNS
            await self._check_global_propagation(health_metrics)
            
            # Store health metrics in CloudWatch
            await self._store_health_metrics(health_metrics)
            
            # Clean up old metrics
            self._cleanup_metrics(health_metrics)
            
            await asyncio.sleep(300)  # Check every 5 minutes
            
        except Exception as e:
            self.logger.error(f"DNS health monitoring error: {e}")
            await asyncio.sleep(60)

    async def _verify_dns_records(self, health_metrics: Dict):
        """Verify integrity of all DNS records"""
        try:
            records = self.route53.list_resource_record_sets(
                HostedZoneId=self.zone_id
            )['ResourceRecordSets']
            
            for record in records:
                response = await self.resolver.query(record['Name'], record['Type'])
                expected_values = self._get_expected_values(record)
                actual_values = [str(r.host) for r in response]
                
                health_metrics['record_integrity'][record['Name']] = {
                    'status': 'matched' if set(expected_values) == set(actual_values) else 'mismatch',
                    'expected': expected_values,
                    'actual': actual_values,
                    'last_check': datetime.datetime.now(datetime.timezone.utc)
                }
                
                if set(expected_values) != set(actual_values):
                    await self._handle_record_mismatch(record, expected_values, actual_values)
                    
        except Exception as e:
            self.logger.error(f"Record verification failed: {e}")
            health_metrics['record_integrity']['status'] = 'error'

    async def _check_global_propagation(self, health_metrics: Dict):
        """Check DNS propagation across global DNS servers"""
        public_dns = [
            '8.8.8.8',  # Google
            '1.1.1.1',  # Cloudflare
            '208.67.222.222',  # OpenDNS
            '9.9.9.9'  # Quad9
        ]
        
        for dns in public_dns:
            resolver = aiodns.DNSResolver(nameservers=[dns])
            try:
                response = await resolver.query(self.domain, 'A')
                health_metrics['propagation_status'][dns] = {
                    'status': 'propagated',
                    'resolved_ip': str(response[0].host),
                    'last_check': datetime.datetime.now(datetime.timezone.utc)
                }
            except Exception as e:
                health_metrics['propagation_status'][dns] = {
                    'status': 'error',
                    'error': str(e),
                    'last_check': datetime.datetime.now(datetime.timezone.utc)
                }

    async def _handle_nameserver_failure(self, nameserver: str):
        """Handle nameserver failures with automatic recovery"""
        self.logger.warning(f"Nameserver failure detected: {nameserver}")
        
        try:
            # Get current nameservers
            current_nameservers = await self._get_route53_nameservers()
            
            # If we have more than minimum required nameservers
            if len(current_nameservers) > 2:
                # Remove failed nameserver
                current_nameservers.remove(nameserver)
                
                # Update GoDaddy nameservers
                credentials = await self._get_godaddy_credentials()
                async with self._create_godaddy_session(credentials) as session:
                    await self._update_godaddy_nameservers(session, current_nameservers)
                    
                # Request new nameserver from Route53
                self.route53.create_reusable_delegation_set()
                
            self.logger.info(f"Nameserver recovery completed for {nameserver}")
            
        except Exception as e:
            self.logger.error(f"Nameserver recovery failed: {e}")

    async def _handle_record_mismatch(self, record: Dict, expected: List[str], actual: List[str]):
        """Handle DNS record mismatches with automatic correction"""
        self.logger.warning(f"Record mismatch detected for {record['Name']}")
        
        try:
            # Create correction change batch
            changes = [{
                'Action': 'UPSERT',
                'ResourceRecordSet': {
                    'Name': record['Name'],
                    'Type': record['Type'],
                    'TTL': record.get('TTL', 300),
                    'ResourceRecords': [{'Value': value} for value in expected]
                }
            }]
            
            # Apply correction
            self.route53.change_resource_record_sets(
                HostedZoneId=self.zone_id,
                ChangeBatch={'Changes': changes}
            )
            
            self.logger.info(f"Record correction applied for {record['Name']}")
            
        except Exception as e:
            self.logger.error(f"Record correction failed: {e}")

    async def _store_health_metrics(self, health_metrics: Dict):
        """Store DNS health metrics in CloudWatch"""
        cloudwatch = self.session.client('cloudwatch')
        
        try:
            # Store response times
            if health_metrics['response_times']:
                cloudwatch.put_metric_data(
                    Namespace='DNS/Health',
                    MetricData=[{
                        'MetricName': 'ResponseTime',
                        'Value': sum(health_metrics['response_times']) / len(health_metrics['response_times']),
                        'Unit': 'Seconds',
                        'Dimensions': [{'Name': 'Domain', 'Value': self.domain}]
                    }]
                )
            
            # Store nameserver health
            healthy_nameservers = sum(
                1 for ns in health_metrics['nameserver_status'].values() 
                if ns['status'] == 'healthy'
            )
            cloudwatch.put_metric_data(
                Namespace='DNS/Health',
                MetricData=[{
                    'MetricName': 'HealthyNameservers',
                    'Value': healthy_nameservers,
                    'Unit': 'Count',
                    'Dimensions': [{'Name': 'Domain', 'Value': self.domain}]
                }]
            )
            
        except Exception as e:
            self.logger.error(f"Failed to store health metrics: {e}")

    def _cleanup_metrics(self, health_metrics: Dict):
        """Clean up old metrics to prevent memory growth"""
        # Keep only last hour of response times
        cutoff = time.time() - 3600
        health_metrics['response_times'] = [
            t for t in health_metrics['response_times']
            if t > cutoff
        ]
        
        # Remove old status entries
        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=24)
        for metric_dict in [health_metrics['nameserver_status'], 
                          health_metrics['record_integrity'],
                          health_metrics['propagation_status']]:
            for key in list(metric_dict.keys()):
                if metric_dict[key].get('last_check', cutoff) < cutoff:
                    del metric_dict[key]