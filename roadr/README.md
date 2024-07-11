# AWS Setup

- Image: Deep Learning OSS Nvidia Driver AMI GPU PyTorch 1.13.1 (Amazon Linux 2)
- GPU: g5.xlarge
- Storage: 300 GB

```
sudo su
cd /usr/local/bin
mkdir ffmpeg && cd ffmpeg
wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
tar -xf ffmpeg-release-amd64-static.tar.xz
cp -a /usr/local/bin/ffmpeg/ffmpeg-6.1-amd64-static/. .
ln -s /usr/local/bin/ffmpeg/ffmpeg /usr/bin/ffmpeg
exit

pip3 install numpy torch transformers pillow scipy torchvision matplotlib torchmetrics pycocotools timm git-python pandas dgl pydantic urllib3==1.26.6

cd ~/.ssh
ssh-keygen -t ed25519 -C "username@domain"
cat id_ed25519.pub

cd ~ && mkdir work && cd work
git clone git@github.com:linqs/jmlr24.git
git clone git@github.com:linqs/psl.git
wget https://dlcdn.apache.org/maven/maven-3/3.9.5/binaries/apache-maven-3.9.5-bin.tar.gz
tar xzvf apache-maven-3.9.5-bin.tar.gz

export JAVA_HOME="/usr/lib/jvm/java-11-openjdk"
export PATH="/usr/bin/java":$PATH
export PATH="/home/ec2-user/work/apache-maven-3.9.5/bin/":$PATH
export LD_LIBRARY_PATH="/home/ec2-user/work/apache-maven-3.9.5/lib/":$LD_LIBRARY_PATH

source ~/.bashrc

cd ~/work/psl/
mvn clean install -D skipTests
./psl-python/.build/force-install.sh

cd /home/ec2-user/work/jmlr24/roadr/scripts
python3 create_data.py
```

