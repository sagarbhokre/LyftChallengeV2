#!/bin/bash
# May need to uncomment and update to find current packages
apt-get update

# Required for demo script! #
pip install scikit-video

# Add your desired packages for each workspace initialization
# Install git lfs to download trained model in the repository
sudo add-apt-repository ppa:git-core/ppa -y
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install

# Clone repository if it does not exist
DIRECTORY=/home/workspace/LyftChallenge
if [ ! -d "$DIRECTORY" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Repository does not exist! Cloning..."
  #git clone https://github.com/sagarbhokre/LyftChallenge.git $DIRECTORY
  #git lfs pull
fi