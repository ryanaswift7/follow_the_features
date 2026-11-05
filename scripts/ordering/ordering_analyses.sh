#!/bin/bash

uv run ftf -c configs/ordering/order_analysis/full_ft.yaml
uv run ftf -c configs/ordering/order_analysis/lr_scheduler.yaml
