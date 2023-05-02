# push new evaluation image to artifact registry
update-evaluation-image:
	gcloud builds submit --config=cloudbuild.yaml \
	--region="global" \
	--substitutions=_LOCATION="us-east1",_REPOSITORY="container-images",_IMAGE="jupyter" .

update-requirements.txt:
	poetry export --format=requirements.txt --without-hashes -o requirements.txt

create-compute:
	gcloud compute instances create-with-container jupyter \
		--project=text2sql-383416 \
		--zone=us-central1-a \
		--machine-type=n1-standard-1 \
		--network-interface=network-tier=PREMIUM,subnet=default \
		--maintenance-policy=TERMINATE \
		--provisioning-model=STANDARD \
		--service-account=84043197426-compute@developer.gserviceaccount.com \
		--scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append \
		--accelerator=count=1,type=nvidia-tesla-v100 \
		--tags=http-server,https-server \
		--image=projects/cos-cloud/global/images/cos-stable-105-17412-1-61 \
		--boot-disk-size=100GB \
		--boot-disk-type=pd-balanced \
		--boot-disk-device-name=jupyter \
		--container-image=us-east1-docker.pkg.dev/text2sql-383416/container-images/jupyter \
		--container-restart-policy=always \
		--no-shielded-secure-boot \
		--shielded-vtpm \
		--shielded-integrity-monitoring \
		--labels=ec-src=vm_add-gcloud,container-vm=cos-stable-105-17412-1-61