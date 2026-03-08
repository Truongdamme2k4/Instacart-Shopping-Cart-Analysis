docker node ls

docker exec -it zerotier-one zerotier-cli join 154a350c86b1c5c7
docker swarm init --advertise-addr 10.229.91.65
docker network create --driver overlay --attachable bigdata_network
docker swarm join --token SWMTKN-1-1v91m11ede216ne6fl897moy1plrs0lia3os9selxrfcqdbfmv-494gxbmbiu501utul13e4yoe6 10.229.91.65:2377
docker stack deploy -c docker-stack.yml hadoop_cluster
docker stack rm hadoop_cluster

docker pull bde2020/hadoop-namenode:2.0.0-hadoop3.2.1-java8
docker pull bde2020/hadoop-datanode:2.0.0-hadoop3.2.1-java8
docker pull bde2020/hadoop-resourcemanager:2.0.0-hadoop3.2.1-java8
docker pull bde2020/hadoop-nodemanager:2.0.0-hadoop3.2.1-java8
docker pull bde2020/spark-master:3.1.1-hadoop3.2
docker pull bde2020/spark-worker:3.1.1-hadoop3.2
docker pull bde2020/spark-history-server:3.1.1-hadoop3.2