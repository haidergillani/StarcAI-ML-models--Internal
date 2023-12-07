from STARC_Cloud_Request import CloudSentimentAnalysis

def entry_point(request):
    sa_interface = CloudSentimentAnalysis()
    return sa_interface.cloud_run(request)