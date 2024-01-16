import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.metrics.pairwise import cosine_similarity
import re


class ContentRecommender:
    # Para simplificar, vou considerar o recebimento to do o dataset já tratado
    # Sendo assim, recebe basicamente um dataset binário com as colunas sendo características e linhas como os items
    def __init__(self, user_char, item_attr, user_item_iter, itemNames=None, userNames=None, metric="cosine"):
        if not sps.isspmatrix_csr(item_attr):
            raise ValueError("The 'item_attr' parameter must be a Scipy CSR Matrix.")

        if not sps.isspmatrix_csr(user_char):
            raise ValueError("The 'user_char' parameter must be a Scipy CSR Matrix.")

        if not sps.isspmatrix_csr(user_item_iter):
            raise ValueError("The 'user_item_iter' parameter must be a Scipy CSR Matrix.")

        if user_item_iter.shape[1] > user_item_iter.shape[0]:
            raise ValueError(
                "The number of rows should be for 'Users', and the number of columns should be for 'Items'.")

        if metric.lower() != "pearson" and metric.lower() != "cosine":
            raise ValueError("The Distance metric must be pearson or cosine.")

        if userNames is None:
            userNames = pd.Series(np.arange(user_item_iter.shape[0]))
        elif len(userNames) != user_item_iter.shape[0]:
            raise ValueError("The size of the users names list needs to be equals to the number of users.")

        if itemNames is None:
            itemNames = pd.Series(np.arange(user_item_iter.shape[1]))
        elif len(itemNames) != user_item_iter.shape[1]:
            raise ValueError("The size of the items names list needs to be equals to the number of items.")

        if not all(isinstance(name, (str, int)) for name in itemNames):
            raise ValueError("The 'itemNames' parameter must be a list containing only strings or integers.")
        if not all(isinstance(name, (str, int)) for name in userNames):
            raise ValueError("The 'userNames' parameter must be a list containing only strings or integers.")
        if user_item_iter.shape[0] != user_char.shape[0] or user_item_iter.shape[1] != item_attr.shape[0] or \
                user_char.shape[1] != item_attr.shape[1]:
            raise ValueError("The Dimensions of the given data are inconsistent.")

        self.user_item_iter = user_item_iter
        self.user_char = user_char
        self.item_attr = item_attr
        self.itemNames = itemNames
        self.userNames = userNames
        self.metric = metric

    def _correlation_pearson_sparse(self, array, matrix):
        """
        Calculate Pearson correlation for sparse matrices.

        Parameters:
        - array (np.ndarray): Numpy array containing only numbers.

        Returns:
        - corr (np.ndarray): Pearson correlation values.
        """
        if not isinstance(array, np.ndarray) or not np.issubdtype(array.dtype, np.number):
            raise ValueError("The 'array' parameter must be a numpy array containing only numbers.")

        yy = array - array.mean()
        xm = matrix.mean(axis=1).A.ravel()
        ys = yy / np.sqrt(np.dot(yy, yy))
        xs = np.sqrt(
            np.add.reduceat(matrix.data ** 2, matrix.indptr[:-1]) - matrix.shape[1] * xm * xm)

        corr = np.add.reduceat(matrix.data * ys[matrix.indices], matrix.indptr[:-1]) / xs
        return corr

    def _search_name(self, search, search_type):
        """
        Search for a name in itemNames/userNames.

        Parameters:
        - search (str or int): Name or index to search for.

        Returns:
        - result (str or int or None): Found name or index, or None if not found.
        """
        if search_type == "User":
            list_to_search = self.userNames
        else:
            list_to_search = self.itemNames
        if search in list_to_search:
            return search
        if isinstance(search, str):
            escaped_search = re.escape(search)
            result = (list_to_search.loc[list_to_search.str.contains(escaped_search)]).reset_index(drop=True)
            if result.empty:
                return None
            return result.loc[0]
        return None

    def _get_recommendation_user(self, user_identification):
        if isinstance(user_identification, (str, np.str_)) or (isinstance(user_identification, int)):
            # Check if the username is in the list and get its index
            found_indices = [index for index, name in enumerate(self.userNames) if name == user_identification]

            if found_indices:
                user_index = found_indices[0]
            else:
                raise ValueError(f"User '{user_identification}' not found.")
        else:
            raise ValueError("Invalid user identification. Provide an integer index or a username.")

        user_movies = self.user_item_iter[user_index, :].toarray()
        user_characteristics = self.user_char[user_index, :]

        unwatched_movies = (user_movies == 0)[0]
        unwatched_movies_charac = self.item_attr[unwatched_movies, :].T

        scores = user_characteristics.dot(unwatched_movies_charac).toarray()[0]

        names = self.itemNames[unwatched_movies]
        order = scores.argsort()[::-1]

        return names.iloc[order], scores[order]

    def get_recommendation(self, to_recommend, N=10, need_numeric=False):
        """
        Get recommendations based on a name/index.

        Parameters:
        - to_recommend: the Name or Index to calculate the recommendation.
        - N: Number of recommendations to return (default is 10).
        - need_numeric: If True, return only the item IDs; if False, return item names and similarity values (default is False).

        Returns:
        - recommendations: Tuple containing ordered item names and similarity values or item IDs and the search_result.
        """
        search_result = self._search_name(to_recommend, "User")

        if search_result is None:
            print(f"{self.recm} not found.")
            return None

        recommendation, values = self._get_recommendation_user(search_result)

        if need_numeric:
            return search_result, recommendation[0:N], values[0:N]

        return search_result, recommendation[0:N]

    def get_items_similars(self, item_identification):
        """
        Get item recommendations based on item identification, public, can be used by the others classes.

        Parameters:
        - item_identification (int or str): Item index or name.

        Returns:
        - Tuple: Ordered item names and similarity values.
        """
        # Adapt item_identification to handle both index and item name
        item_identification = self._search_name(item_identification, "Item")

        if isinstance(item_identification, int):
            item_index = item_identification
        elif isinstance(item_identification, (str, np.str_)):
            # Check if the item name is in the list and get its index
            # I dont know why but if item_id in itemNames dont work
            found_indices = [index for index, name in enumerate(self.itemNames) if name == item_identification]

            if found_indices:
                item_index = found_indices[0]
            else:
                print(f"Item '{item_identification}' not found.")
                raise ValueError("Item not found.")
        else:
            raise ValueError("Invalid item identification. Provide an integer index or an item name.")

        item = self.item_attr[item_index, :].toarray().ravel()

        if self.metric == "cosine":
            similarity = cosine_similarity(item.reshape(1, -1), self.item_attr)[0]
        elif self.metric == "pearson":
            similarity = self._correlation_pearson_sparse(item, self.item_attr)
        else:
            raise ValueError("Invalid metric. Choose 'cosine' or 'pearson'.")

        # order = np.argsort(-similarity)[1:]

        order = np.argsort(-similarity)[1:]

        top_10 = similarity[order]
        top_10_names = self.itemNames[order]

        return item_identification, top_10_names, top_10

    def get_user_items(self, user_identification):
        search_result = self._search_name(user_identification, "User")

        if search_result is None:
            print(f"User not found.")
            return None

        user_index = (self.userNames == user_identification).idxmax()

        first_user = self.user_item_iter[user_index, :].toarray().ravel()
        user_items = first_user > 0
        user_items_names = self.itemNames[user_items]

        # Ordenar user_items_names de acordo com os valores correspondentes em first_user
        sorted_indices = np.argsort(first_user[user_items])[::-1]
        user_items_names = user_items_names.iloc[sorted_indices]

        return user_items_names
