import json, boto3, os, datetime, logging, sys, re
from decimal import Decimal
import pandas as pd
import numpy as np
import itertools
import time

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    # Dump the event for creating a test later
    logger.info(json.dumps(event))

    # Get the s3 bucket name from the full path
    version = event['Input']['Payload']['dataBucketPath'].split('/')[3]
    logger.info(version)
    suffix = '/'+version+'/train'
    logger.info(suffix)
    dataBucket = event['Input']['Payload']['dataBucketPath'].split('/')[2]
    logger.info(dataBucket)

    # Set fixed locations to expect validation data to exist
    objectPath = version+'/validation/'
    fileName = 'iris.csv'
    
    s3 = boto3.resource('s3')
    sagemaker = boto3.client('runtime.sagemaker')
    #thisBucket = s3.Bucket(dataBucket)
    
    

    # Download the validation file to use for testing
    try:
      key = objectPath+fileName
      s3.Bucket(dataBucket).download_file(key, '/tmp/iris.csv')
      print("file downloaded")
    except:
      e = sys.exc_info()[0]
      f = sys.exc_info()[1]
      g = sys.exc_info()[2]
      logger.error("error (update error): "+str(e) + str(f) + str(g))
    
    # Read Data and create test data
    shape=pd.read_csv("/tmp/iris.csv", header=None)
    
    a = [15*i for i in range(2)]
    b = [15+i for i in range(10)]
    indices = [i+j for i,j in itertools.product(a,b)]
    
    test_data = shape.drop(shape.columns[[0]],axis=1)
    test_data = test_data.iloc[indices]
    test_data_with_label = shape.iloc[indices]
    
    test_data.to_csv("/tmp/data-test.csv",index=False,header=False)
    test_data_with_label.to_csv("/tmp/data-test-label.csv",index=False,header=False)
    
    test_data = shape.iloc[indices[:-1]]
    test_data.to_csv('/tmp/test_data.csv')

    logger.info(event['Input']['Payload']['Endpoint'])
    endpointName = endpoint=event['Input']['Payload']['Endpoint']



    try:
      predictions_a = ""
      with open('/tmp/data-test.csv', 'r') as f:
        for row in f:
            print(".", end="", flush=True)
            payload = row.rstrip('\n')
            response = sagemaker.invoke_endpoint(EndpointName=endpointName,
                                       ContentType="text/csv",
                                       Body=payload,
                                       TargetVariant='Variant1'
                                       )
            predictions_a = ','.join([predictions_a, response['Body'].read().decode('utf-8')])
            time.sleep(0.1)

      predictions_a = predictions_a.replace('\n','')
      predictions_a = predictions_a.split(",")
      predictions_a.pop(0)
      print(predictions_a)
      print("Done!")
    except:
        raise

    print("---- predictions_a ----")
    print(predictions_a)

    labels = test_data_with_label[0].to_numpy()
    preds_a = np.array(predictions_a)
    preds_a = preds_a.astype(np.int)
    # Calculate the model accuracy
    accuracy_a = np.count_nonzero(preds_a == labels) / len(labels)
    print("accuracy: ", accuracy_a)
    
    
    try:
      predictions_b = ""
      with open('/tmp/data-test.csv', 'r') as f:
        for row in f:
            print(".", end="", flush=True)
            payload = row.rstrip('\n')
            response = sagemaker.invoke_endpoint(EndpointName=endpointName,
                                       ContentType="text/csv",
                                       Body=payload,
                                       TargetVariant='Variant2'
                                       )
            predictions_b = ','.join([predictions_b, response['Body'].read().decode('utf-8')])
            time.sleep(0.1)

      predictions_b = predictions_b.replace('\n','')
      predictions_b = predictions_b.split(",")
      predictions_b.pop(0)
      print(predictions_b)
      print("Done!")
    except:
        raise

    print("---- predictions_b ----")
    print(predictions_b)

    labels = test_data_with_label[0].to_numpy()
    preds_b = np.array(predictions_b)
    preds_b = preds_b.astype(np.int)
    # Calculate the model accuracy
    accuracy_b = np.count_nonzero(preds_b == labels) / len(labels)
    print("accuracy: ", accuracy_b)


    # Update DynamoDB Table with accuracy value
    dynamo = boto3.resource('dynamodb')
    table = dynamo.Table(event['Input']['Payload']['dynamodb'])

    #Update accuracy for ModelA
    try:
      response = table.update_item(
        Key={
          'RunId': event['Input']['Payload']['JobA']
        },
        UpdateExpression="set Accuracy = :a",
        ExpressionAttributeValues={
          ':a': Decimal(str(accuracy_a))
        },
        ReturnValues="UPDATED_NEW"
      )
      print("-----")
      logger.debug(json.dumps(response))
    except:
      e = sys.exc_info()[0]
      f = sys.exc_info()[1]
      g = sys.exc_info()[2]
      logger.error("error (update error): "+str(e) + str(f) + str(g))

    #Update accuracy for ModelB
    try:
      response = table.update_item(
        Key={
          'RunId': event['Input']['Payload']['JobB']
        },
        UpdateExpression="set Accuracy = :a",
        ExpressionAttributeValues={
          ':a': Decimal(str(accuracy_b))
        },
        ReturnValues="UPDATED_NEW"
      )
      print("-----")
      logger.debug(json.dumps(response))
    except:
      e = sys.exc_info()[0]
      f = sys.exc_info()[1]
      g = sys.exc_info()[2]
      logger.error("error (update error): "+str(e) + str(f) + str(g))