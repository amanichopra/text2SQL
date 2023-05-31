# Text2SQL

This is the capstone project for Columbia's MS in Data Science degree where we partner with L'Oreal, the world's largest cosmetics company that strives to offer everyone the best of beauty \cite{b1}, to build a text-to-SQL system that makes it easier for non-technical employees to query databases through natural language, rather than crafting SQL queries. There is significant research already done that uses pre-trained LLMs (large language models) to fine-tune on custom text-to-SQL datasets like Spider and CoSQL to achieve state-of-the-art performance on this task. We will train & benchmark these models, explore fine-tuning on L'Oreal's internal data, and aim to deploy a production-grade system that can be used to drive efficiency and reduce cost. Our final report can be accessed [here](https://drive.google.com/file/d/1PX6vUI4bwIkYEefF84A6gFM1KoPiy0To/view?usp=sharing).

Contributors: Aman Chopra, Zicheng Huang, Jingru Chen

## Environment Set Up

Follow the below steps to set up your environment before running any code.

### Local

1. Install poetry via pip: `pip3 install poetry`.
2. Install all dependendencies via `poetry install`. This will automatically install required packages specified in the `poetry.lock` file.
3. Replace the paths in the `.env` file with your paths to the [Spider](https://drive.google.com/uc?export=download&id=1TqleXec_OykOYFREKKtschzY29dUcVAQ) and [CoSQL](https://drive.google.com/uc?export=download&id=1Y3ydpFiQQ3FC0bzdfy3groV95O_f1nXF) datasets.

### Cloud

*Note that you can replace steps 1 and 2 if you use the Makefile to trigger a GCP Cloudbuild run to build and push the image for you. Use `make update-evaluation-image`.*

1. Build the docker image using the `Dockerfile` in the root directory. 
2. Push the image to a registry like GCP Artifict Registry. 
3. Spin up a VM (GCE on GCP) or managed notebook (Vertex AI on GCP) with the image URI as a startup requirement.
