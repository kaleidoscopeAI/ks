#!/usr/bin/env python3
import boto3
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import datetime
import json
from dataclasses import dataclass
import logging
from scipy import stats
import time

@dataclass
class MetricThresholds:
    response_time_p95: float = 1.0
    availability_min: float = 99.9
    error_rate_max: float = 0.1
    latency_anomaly_std: float = 2.0
    ssl_expiry_min_days: int = 30

class DomainMetricsProcessor:
    def __init__(self, domain: str, region: str = "us-east-2"):
        self.domain = domain
        self.region = region
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        self.dynamodb = boto3.client('dynamodb', region_name=region)
        self.s3 = boto3.client('s3', region_name=region)
        self.thresholds = MetricThresholds()
        self.logger = self._setup_logger()
        self._initialize_storage()

    def _setup_logger(self):
        logger = logging.getLogger('DomainMetricsProcessor')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(handler)
        return logger

    def _initialize_storage(self):
        self._create_metrics_table()
        self._create_metrics_bucket()

    def _create_metrics_table(self):
        try:
            self.dynamodb.create_table(
                TableName=f'domain-metrics-{self.domain.replace(".", "-")}',
                KeySchema=[
                    {'AttributeName': 'metric_name', 'KeyType': 'HASH'},
                    {'AttributeName': 'timestamp', 'KeyType': 'RANGE'}
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'metric_name', 'AttributeType': 'S'},
                    {'AttributeName': 'timestamp', 'AttributeType': 'S'}
                ],
                BillingMode='PAY_PER_REQUEST'
            )
        except self.dynamodb.exceptions.ResourceInUseException:
            pass

    def _create_metrics_bucket(self):
        try:
            self.s3.create_bucket(
                Bucket=f'domain-metrics-{self.domain.replace(".", "-")}',
                CreateBucketConfiguration={'LocationConstraint': self.region}
            )
        except self.s3.exceptions.BucketAlreadyExists:
            pass

    def process_metrics(self, start_time: datetime.datetime, 
                       end_time: datetime.datetime) -> Dict:
        raw_metrics = self._get_raw_metrics(start_time, end_time)
        processed_metrics = self._calculate_metrics(raw_metrics)
        alerts = self._check_thresholds(processed_metrics)
        anomalies = self._detect_anomalies(raw_metrics)
        
        results = {
            'metrics': processed_metrics,
            'alerts': alerts,
            'anomalies': anomalies,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        self._store_results(results)
        return results

    def _get_raw_metrics(self, start_time: datetime.datetime, 
                        end_time: datetime.datetime) -> Dict[str, List]:
        metrics = {}
        
        # Get metrics from CloudWatch
        for metric_name in ['HTTP/ResponseTime', 'Global/Availability', 'SSL/DaysToExpiry']:
            response = self.cloudwatch.get_metric_data(
                MetricDataQueries=[{
                    'Id': 'metric',
                    'MetricStat': {
                        'Metric': {
                            'Namespace': 'Domain/Health',
                            'MetricName': metric_name,
                            'Dimensions': [{'Name': 'Domain', 'Value': self.domain}]
                        },
                        'Period': 60,
                        'Stat': 'Average'
                    }
                }],
                StartTime=start_time,
                EndTime=end_time
            )
            
            metrics[metric_name] = {
                'timestamps': response['MetricDataResults'][0]['Timestamps'],
                'values': response['MetricDataResults'][0]['Values']
            }
            
        return metrics

    def _calculate_metrics(self, raw_metrics: Dict) -> Dict:
        results = {}
        
        for metric_name, data in raw_metrics.items():
            values = np.array(data['values'])
            results[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'p50': float(np.percentile(values, 50)),
                'p95': float(np.percentile(values, 95)),
                'p99': float(np.percentile(values, 99))
            }
            
        return results

    def _check_thresholds(self, metrics: Dict) -> List[Dict]:
        alerts = []
        
        # Response Time Alerts
        if metrics['HTTP/ResponseTime']['p95'] > self.thresholds.response_time_p95:
            alerts.append({
                'severity': 'high',
                'metric': 'response_time',
                'message': f'P95 response time ({metrics["HTTP/ResponseTime"]["p95"]:.2f}s) exceeds threshold'
            })
            
        # Availability Alerts
        if metrics['Global/Availability']['mean'] < self.thresholds.availability_min:
            alerts.append({
                'severity': 'critical',
                'metric': 'availability',
                'message': f'Availability ({metrics["Global/Availability"]["mean"]:.1f}%) below threshold'
            })
            
        # SSL Certificate Alerts
        if metrics['SSL/DaysToExpiry']['min'] < self.thresholds.ssl_expiry_min_days:
            alerts.append({
                'severity': 'warning',
                'metric': 'ssl',
                'message': f'SSL certificate expires in {int(metrics["SSL/DaysToExpiry"]["min"])} days'
            })
            
        return alerts

    def _detect_anomalies(self, raw_metrics: Dict) -> List[Dict]:
        anomalies = []
        
        for metric_name, data in raw_metrics.items():
            values = np.array(data['values'])
            timestamps = data['timestamps']
            
            # Z-score anomaly detection
            z_scores = stats.zscore(values)
            anomaly_indices = np.where(np.abs(z_scores) > self.thresholds.latency_anomaly_std)[0]
            
            for idx in anomaly_indices:
                anomalies.append({
                    'metric': metric_name,
                    'timestamp': timestamps[idx].isoformat(),
                    'value': float(values[idx]),
                    'z_score': float(z_scores[idx])
                })
            
        return anomalies

    def _store