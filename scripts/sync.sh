#!/bin/bash
rsync -alPvz -e 'ssh -p 2222' . $1@cf.rmi.yaht.net:/home/$1/fractalogy
