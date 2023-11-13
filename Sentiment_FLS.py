Data1 = [{'sentence': 'We expected economic weakness in some emerging markets.', 'Positive': 0, 'Neutral': 1, 'Negative': 2, 'label_sentiment': 'A'}, {'sentence': 'This turned out to have a significantly greater impact than we had projected.', 'Positive': 3, 'Neutral': 4, 'Negative': 5, 'label_sentiment': 'B'}]

Data2 = [{'sentence': 'We expected economic weakness in some emerging markets.', 'Specific FLS': 6, 'non-specific FLS': 7, 'not FLS': 8, 'label_fls': 'C'}, {'sentence': 'This turned out to have a significantly greater impact than we had projected.', 'Specific FLS': 9, 'non-specific FLS': 10, 'not FLS': 11, 'label_fls': 'D'}]

Result = [{'sentence': 'We expected economic weakness in some emerging markets.', 'Positive': 0, 'Neutral': 1, 'Negative': 2, 'label_sentiment': 'A', 'Specific FLS': 6, 'non-specific FLS': 7, 'not FLS': 8, 'label_fls': 'C'}, {'sentence': 'This turned out to have a significantly greater impact than we had projected.', 'Positive': 3, 'Neutral': 4, 'Negative': 5, 'label_sentiment': 'B', 'Specific FLS': 9, 'non-specific FLS': 10, 'not FLS':11, 'label_fls': 'D'}]

def merge_data(list1, list2):
    # Merge corresponding dictionaries from both lists
    return [{**d1, **d2} for d1, d2 in zip(list1, list2)]

# Test the function with the provided data
merged_result = merge_data(Data1, Data2)
print(merged_result)
# Check if merged_result matches your expected Result
assert(merged_result==Result)
