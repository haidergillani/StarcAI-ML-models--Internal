from STARC_Cloud_Request import CloudSentimentAnalysis

from google.cloud import storage
import os
import nltk

def download_blob(bucket_name, source_blob_name, destination_dir):
    """Downloads a blob from the specified bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=source_blob_name)  # Get list of files in the folder

    for blob in blobs:
        file_path = os.path.join(destination_dir, os.path.basename(blob.name))
        blob.download_to_filename(file_path)
        

# Use this function to download data at the start of your Cloud Function
nltk_data_path = '/tmp/nltk_data'  # Temporary directory in Cloud Function environment
os.makedirs(nltk_data_path, exist_ok=True)


bucket_name = 'nltk_data_starcai'
source_blob_name = 'punkt'

# This will download punkt.zip from the specified path in your bucket
download_blob(bucket_name, source_blob_name, nltk_data_path)

# Set NLTK to use the downloaded data
nltk.data.path.append(nltk_data_path)



# The following line is for local testing only
# When deployed to Google Cloud Functions, the entry_point function will be called with the HTTP request object
if __name__ == "__main__":
    user_data = 'We expected economic weakness in some emerging markets. This turned out to have a significantly greater impact than we had projected. Based on these estimates, our revenue will be lower than our original guidance for the quarter, with other items remaining broadly in line with our guidance. As we exit a challenging quarter, we may still find a way to retain the strength of our business. We use periods of adversity to re-examine our approach and use our flexibility, adaptability, and creativity to emerge better afterward.'
    sa_interface = CloudSentimentAnalysis()
    # Return model's response
    sa_interface.cloud_run(user_data)    
    # entry_point(user_data)
    
