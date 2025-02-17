await self._store_metric('HTTP/ResponseTime', response_time)
                        await self._store_metric('HTTP/Status', status)
                        
                        if response_time > self.config.threshold_response_time:
                            await self._handle_warning('http', f'Response time ({response_time:.2f}s) exceeds threshold')
                        
                        if status != 200:
                            await self._handle_error('http', f'Non-200 status code: {status}')
                            
                except Exception as e:
                    self.logger.error(f"HTTP monitoring error: {e}")
                    await self._handle_error('http', str(e))
                    
                await asyncio.sleep(self.config.check_interval)

    async def _monitor_global_availability(self):
        while True:
            try:
                results = await asyncio.gather(*[
                    self._check_region_availability(region)
                    for region in self.config.regions_to_check
                ])
                
                availability = sum(1 for r in results if r['status'] == 'available') / len(results) * 100
                self.metrics['global_availability'] = availability
                
                await self._store_metric('Global/Availability', availability)
                
                if availability < self.config.threshold_availability:
                    await self._handle_warning('availability', 
                        f'Global availability ({availability:.1f}%) below threshold')
                    
                # Store regional metrics
                for result in results:
                    await self._store_metric(
                        'Regional/ResponseTime',
                        result['response_time'],
                        {'Region': result['region']}
                    )
                    
            except Exception as e:
                self.logger.error(f"Global availability monitoring error: {e}")
                await self._handle_error('availability', str(e))
                
            await asyncio.sleep(self.config.check_interval * 2)

    async def _check_region_availability(self, region: str) -> Dict:
        start_time = time.time()
        try:
            session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            )
            async with session:
                async with session.get(
                    f'https://{self.config.domain}',
                    headers={'X-Region-Check': region},
                    ssl=self.ssl_context
                ) as response:
                    response_time = time.time() - start_time
                    return {
                        'region': region,
                        'status': 'available' if response.status == 200 else 'error',
                        'response_time': response_time,
                        'status_code': response.status
                    }
        except Exception as e:
            return {
                'region': region,
                'status': 'error',
                'response_time': time.time() - start_time,
                'error': str(e)
            }

    async def _store_metric(self, name: str, value: float, dimensions: Dict = None):
        cloudwatch = self.session.client('cloudwatch')
        metric_data = {
            'MetricName': name,
            'Value': value,
            'Unit': 'Seconds' if 'ResponseTime' in name else 'None',
            'Dimensions': [
                {'Name': 'Domain', 'Value': self.config.domain}
            ]
        }
        
        if dimensions:
            metric_data['Dimensions'].extend([
                {'Name': k, 'Value': v} for k, v in dimensions.items()
            ])
            
        try:
            cloudwatch.put_metric_data(
                Namespace='Domain/Health',
                MetricData=[metric_data]
            )
        except Exception as e:
            self.logger.error(f"Failed to store metric {name}: {e}")

    async def _handle_error(self, component: str, error: str):
        sns = self.session.client('sns')
        error_topic = self._get_or_create_sns_topic('domain-health-errors')
        
        try:
            sns.publish(
                TopicArn=error_topic,
                Subject=f"Domain Health Error - {self.config.domain}",
                Message=json.dumps({
                    'domain': self.config.domain,
                    'component': component,
                    'error': error,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'metrics': self.metrics
                }, indent=2)
            )
        except Exception as e:
            self.logger.error(f"Failed to publish error notification: {e}")

    async def _handle_warning(self, component: str, message: str):
        sns = self.session.client('sns')
        warning_topic = self._get_or_create_sns_topic('domain-health-warnings')
        
        try:
            sns.publish(
                TopicArn=warning_topic,
                Subject=f"Domain Health Warning - {self.config.domain}",
                Message=json.dumps({
                    'domain': self.config.domain,
                    'component': component,
                    'warning': message,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'metrics': self.metrics
                }, indent=2)
            )
        except Exception as e:
            self.logger.error(f"Failed to publish warning notification: {e}")

    def _get_or_create_sns_topic(self, name: str) -> str:
        sns = self.session.client('sns')
        try:
            topic = sns.create_topic(Name=name)
            return topic['TopicArn']
        except Exception as e:
            self.logger.error(f"Failed to get/create SNS topic: {e}")
            raise

    async def _get_nameservers(self) -> List[str]:
        try:
            response = await self.resolver.query(self.config.domain, 'NS')
            return [str(ns.host) for ns in response]
        except Exception as e:
            self.logger.error(f"Failed to get nameservers: {e}")
            raise

    async def _get_ssl_certificate(self) -> Dict:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        
        try:
            conn = await asyncio.open_connection(
                self.config.domain, 443, ssl=ssl_context
            )
            _, writer = conn
            cert = writer.get_extra_info('ssl_object').getpeercert()
            writer.close()
            await writer.wait_closed()
            
            return {
                'Subject': dict(x[0] for x in cert['subject']),
                'Issuer': dict(x[0] for x in cert['issuer']),
                'NotBefore': cert['notBefore'],
                'NotAfter': cert['notAfter'],
                'SerialNumber': cert['serialNumber']
            }
        except Exception as e:
            self.logger.error(f"Failed to get SSL certificate: {e}")
            raise

async def main():
    config = MonitoringConfig(
        domain="artificialthinker.com",
        check_interval=30,
        threshold_response_time=0.5,
        threshold_availability=99.9,
        ssl_expiry_warning=30,
        regions_to_check=[
            'us-east-1', 'us-west-2', 'eu-west-1', 
            'ap-southeast-1', 'ap-northeast-1'
        ]
    )
    
    monitor = DomainHealthMonitor(config)
    await monitor.start_monitoring()

if __name__ == "__main__":
    asyncio.run(main())