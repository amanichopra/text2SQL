# push new evaluation image to artifact registry
update-evaluation-image:
	gcloud builds submit --config=cloudbuild.yaml \
	--region="global" \
	--substitutions=_LOCATION="us-east1",_REPOSITORY="container-images",_IMAGE="jupyter" .

update-requirements.txt:
	poetry export --format=requirements.txt --without-hashes -o requirements.txt