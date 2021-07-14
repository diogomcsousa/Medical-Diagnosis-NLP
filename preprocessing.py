from pymongo import MongoClient, ASCENDING
from datetime import datetime
from data_exploration.exploration import visualise_stock_data, visualise_oil_data, visualise_pandemic_data

import pandas as pd

import numpy as np


def stock_data_clean(db):
    file = pd.read_csv('Datasets/AAL_data.csv')

    # Features to keep after clean
    keep_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    new_file = file[keep_cols]
    new_file['Date'] = [datetime.strptime(g, '%Y-%m-%d') for g in file['Date']]

    new_file.to_csv('Datasets/AAL_processed.csv', index=False)
    visualise_stock_data(new_file)

    aal_stock_price_updated = db.aal_stock_price_processed
    aal_stock_price_updated.create_index([('Date', ASCENDING)], unique=True)

    for row in new_file.iterrows():
        document_updated = {
            'Date': row[1]['Date'],
            'Open': row[1]['Open'],
            'High': row[1]['High'],
            'Low': row[1]['Low'],
            'Close': row[1]['Close'],
            'Volume': row[1]['Volume'],
        }
        # Insert cleaned document
        aal_stock_price_updated.insert_one(document_updated)


def oil_data_clean(db):
    file = pd.read_csv('Datasets/oil_price_data.csv')

    # Select features to keep
    keep_cols = ['Date', 'Price']
    new_file = file[keep_cols]
    new_file['Date'] = [datetime.strptime(g, '%b %d, %Y') for g in file['Date']]

    new_file.to_csv('Datasets/oil_price_processed.csv', index=False)

    # Perform visualisation of data to clean
    visualise_oil_data(new_file)

    oil_price_updated = db.oil_price_processed
    oil_price_updated.create_index([('Date', ASCENDING)], unique=True)

    for row in new_file.iterrows():
        document_updated = {
            'Date': row[1]['Date'],
            'Price': row[1]['Price'],
        }
        # Insert cleaned document
        oil_price_updated.insert_one(document_updated)


def pandemic_data_clean(db):
    file = pd.read_csv('Datasets/pandemic_data.csv')
    pandemic_data_renamed = file.rename(columns={'date': 'Date', 'positive': 'New Cases', 'death': 'Deaths'})

    # Select features to keep
    keep_cols = ['Date', 'New Cases', 'Deaths']
    new_file = pandemic_data_renamed[keep_cols]
    new_file['Date'] = [datetime.strptime(str(g), '%Y%m%d') for g in pandemic_data_renamed['Date']]

    new_file.fillna(0, inplace=True)

    pandemic_updated = db.pandemic_processed
    pandemic_updated.create_index([('Date', ASCENDING)], unique=True)

    for row in new_file.iterrows():
        if row[1]['Date'] <= datetime.strptime('20200531', '%Y%m%d'):
            document_updated = {
                'Date': row[1]['Date'],
                'New Cases': row[1]['New Cases'],
                'Deaths': row[1]['Deaths']
            }
            # Insert cleaned document
            pandemic_updated.insert_one(document_updated)
        else:
            # Remove observation outside the desired span
            new_file.drop(row[0], inplace=True)

    new_file.to_csv('Datasets/pandemic_processed.csv', index=False)
    visualise_pandemic_data(new_file)


def data_integration(db):
    aal_data = pd.read_csv('Datasets/AAL_processed.csv')
    oil_data = pd.read_csv('Datasets/oil_price_processed.csv')
    pandemic_data = pd.read_csv('Datasets/pandemic_processed.csv')

    integrate_oil = aal_data.merge(oil_data, on='Date', how='left')
    integrate_pandemic = integrate_oil.merge(pandemic_data, on='Date', how='left')

    # Select the features to integrate
    integrate_all = integrate_pandemic[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Price', 'New Cases', 'Deaths']]

    integrate_all['Date'] = [datetime.strptime(str(g), '%Y-%m-%d') for g in integrate_all['Date']]
    integrate_all.fillna(0, inplace=True)
    integrate_all.to_csv('Datasets/integrated_data_processed.csv')

    integrated_data_processed = db.integrated_data_processed
    integrated_data_processed.create_index([('Date', ASCENDING)], unique=True)

    for row in integrate_all.iterrows():
        document_updated = {
            'Date': row[1]['Date'],
            'Open': row[1]['Open'],
            'High': row[1]['High'],
            'Low': row[1]['Low'],
            'Close': row[1]['Close'],
            'Volume': row[1]['Volume'],
            'Price': row[1]['Price'],
            'New Cases': row[1]['New Cases'],
            'Deaths': row[1]['Deaths'],
        }
        # Insert cleaned integrated document
        integrated_data_processed.insert_one(document_updated)


def transform_data(db):
    data = pd.read_csv('Datasets/integrated_data_processed.csv')
    transform = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Price', 'New Cases', 'Deaths']]

    # Perform normalisation on integrated data from all data sources
    transformed_data = ((transform - transform.min()) / (transform.max() - transform.min()))
    transformed_data.to_csv('Datasets/integrated_data_transformed.csv')

    transformed_data_collection = db.transformed_data

    for row in transformed_data.iterrows():
        document_updated = {
            'Open': row[1]['Open'],
            'High': row[1]['High'],
            'Low': row[1]['Low'],
            'Close': row[1]['Close'],
            'Volume': row[1]['Volume'],
            'Price': row[1]['Price'],
            'New Cases': row[1]['New Cases'],
            'Deaths': row[1]['Deaths'],
        }
        # Insert transformed integrated document
        transformed_data_collection.insert_one(document_updated)


def data_clean(db):
    stock_data_clean(db)
    oil_data_clean(db)
    pandemic_data_clean(db)
    data_integration(db)
    transform_data(db)

