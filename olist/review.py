import pandas as pd
import numpy as np
import math
from olist.data import Olist
from olist.order import Order


class Review:

    def __init__(self):
        # Import data only once
        olist = Olist()
        self.data = olist.get_data()
        self.order = Order()

    def get_review_length(self):
        """
        Returns a DataFrame with:
       'review_id', 'length_review', 'review_score'
        """
        reviews = self.data['order_reviews'].copy()

        reviews['review_comment_message'] = reviews['review_comment_message'].fillna('')
        reviews['length_review'] = reviews['review_comment_message'].apply(len)

        return reviews[['review_id', 'length_review', 'review_score']]

    def get_main_product_category(self):
        """
        Returns a DataFrame with:
       'review_id', 'order_id','product_category_name'
        """
        reviews = self.data['order_reviews'][['review_id', 'order_id']].copy()
        order_items = self.data['order_items'][['order_id', 'product_id']].copy()
        products = self.data['products'][['product_id', 'product_category_name']].copy()

        df = reviews.merge(order_items, on='order_id')
        df = df.merge(products, on='product_id')

        # Bir siparişte birden fazla ürün/kategori olabilir, ilkini alalım
        df = df.drop_duplicates(subset='review_id', keep='first')

        return df[['review_id', 'order_id', 'product_category_name']]

    def get_training_data(self):
        """
        Returns a DataFrame with:
        'review_id', 'order_id', 'length_review', 'review_score', 'product_category_name'
        """
        review_length = self.get_review_length()
        main_category = self.get_main_product_category()

        training_set = main_category.merge(review_length, on='review_id')

        return training_set