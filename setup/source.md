### Setup Postgres

We install postgres and its dev tools (necessary to build lantern from source). We also start postgres, and set up a user `postgres` with password `postgres` and create a database called `ourdb`

```bash
# Install postgres and its dev tools
sudo apt-get -y -qq update
sudo apt-get -y -qq install postgresql postgresql-server-dev-all

# Start postgres
sudo service postgresql start

# Create user, password, and db
sudo -u postgres psql -U postgres -c "ALTER USER postgres PASSWORD 'postgres'"
sudo -u postgres psql -U postgres -c 'DROP DATABASE IF EXISTS ourdb'
sudo -u postgres psql -U postgres -c 'CREATE DATABASE ourdb'
```

### Install Lantern from source

```bash
git clone --recursive https://github.com/lanterndata/lantern.git

cd lantern
mkdir build
cd build
pwd
cmake ..
make install

sudo -u postgres psql -U postgres -c 'CREATE EXTENSION lantern'
```
