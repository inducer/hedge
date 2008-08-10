#! /bin/sh

rsync --progress --delete --recursive .build/html/* tiker.net:public_html/doc/hedge
