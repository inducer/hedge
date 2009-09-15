#! /bin/sh

rsync --progress --delete --recursive dev-reference/* tiker.net:public_html/doc/hedge-reference
