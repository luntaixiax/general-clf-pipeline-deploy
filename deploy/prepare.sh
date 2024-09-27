# create k8s secret for secrets.toml (at root folder)
cd ..
kubectl create namespace general-clf
kubectl create secret generic vault-secret \
--from-file=secrets.toml \
--namespace=general-clf