#! /bin/sh

rsync --progress --delete --recursive user-reference/* tiker.net:public_html/doc/hedge
