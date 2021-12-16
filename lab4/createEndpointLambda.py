import boto3
import os
import sagemaker
import time
import random
import uuid
import logging
import io
import random
import json
import sys
from datetime import datetime, timedelta

from sagemaker import image_uris
from sagemaker import session
from sagemaker.session import production_variant

sm_session = session.Session(boto3.Session())
region = boto3.Session().region_name

sagemaker_role = os.environ['sagemakerRole']

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

model_namea = f"DEMO-decission-tree-pred-{datetime.now():%Y-%m-%d-%H-%M-%S}"
model_nameb = f"DEMO-random-forest-pred-{datetime.now():%Y-%m-%d-%H-%M-%S}"
model_bucketA = os.environ['modelBucketA']
model_bucketB = os.environ['modelBucketB']
dynamodb = os.environ['dynamoDBTable']


def lambda_handler(event, context):
    # Dump the event for creating a test later
    logger.info(json.dumps(event))
    logger.info(sagemaker_role)
    logger.info(model_namea)
    logger.info(model_nameb)
    
    ecr_ArnA = event['Input'][0]['ecrArnA']
    ecr_ArnB = event['Input'][0]['ecrArnB']
    
    JobIDA = event['Input'][0]['ModelA']
    JobIDB = event['Input'][0]['ModelB']
    dataBucket = event['Input'][0]['dataBucketPath']
    endpoint=event['Input'][0]['Endpoint']
    
    #endpoint=event['Input']['Endpoint']
    #endpoint=f"Iris-EndPoint-{datetime.now():%Y-%m-%d-%H-%M-%S}"
    
    
    sm_session.create_model(name=model_namea, role=sagemaker_role, container_defs={
        'Image': ecr_ArnA,
        'ModelDataUrl': '{}/{}/output/model.tar.gz'.format(model_bucketA,'JobA-'+event['Input'][0]['BuildId'])
    })
    sm_session.create_model(name=model_nameb, role=sagemaker_role, container_defs={
        'Image': ecr_ArnB,
        'ModelDataUrl': '{}/{}/output/model.tar.gz'.format(model_bucketB,'JobB-'+event['Input'][0]['BuildId'])
    })
    
    # Create a step to generate an Amazon SageMaker endpoint configuration
    
    varianta = production_variant(model_name=model_namea,
                                  instance_type="ml.m5.xlarge",
                                  initial_instance_count=1,
                                  variant_name='Variant1',
                                  initial_weight=1)
    variantb = production_variant(model_name=model_nameb,
                                  instance_type="ml.m5.xlarge",
                                  initial_instance_count=1,
                                  variant_name='Variant2',
                                  initial_weight=1)
    
    sm_session.endpoint_from_production_variants(
    name=endpoint,
    production_variants=[varianta, variantb],
    wait=False
    )
    
    return {
        'statusCode': 200,
        'body': json.dumps('Endpoint creation started!'),
        'Endpoint': endpoint,
        'JobA': JobIDA,
        'JobB': JobIDB,
        'dynamodb': dynamodb,
        'dataBucketPath': dataBucket
        }