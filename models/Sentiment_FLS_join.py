class DataMerger:
    @staticmethod
    def merge_data(list1, list2):
        if list1 is None or list2 is None:
            raise ValueError("No results to process.")
        # Merge corresponding dictionaries from both lists
        return [{**d1, **d2} for d1, d2 in zip(list1, list2)]