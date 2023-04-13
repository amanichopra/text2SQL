# push new evaluation image to artifact registry
update-evaluation-image:
	gcloud builds submit --config=cloudbuild.yaml \
	--region="us-central1" \ 
	--substitutions=_LOCATION="us-central1",_REPOSITORY="container-images",_IMAGE="jupyter" .