#!/bin/sh

P_ID="$1"

while True
do
    sleep 600
    kill -STOP $P_ID
    sleep 0.5
    kill -CONT $P_ID
done
