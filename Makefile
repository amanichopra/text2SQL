# push new evaluation image to artifact registry
update-evaluation-image:
	gcloud builds submit --config=cloudbuild.yaml \
	--region="us-east1" \ 
	--substitutions=_LOCATION="us-east1",_REPOSITORY="container-images",_IMAGE="evaluation" .