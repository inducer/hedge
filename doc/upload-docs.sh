#! /bin/sh

rsync --progress --delete --recursive user-reference/* tikernet@tiker.net:public_html/doc/hedge
