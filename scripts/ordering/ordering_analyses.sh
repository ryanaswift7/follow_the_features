#!/bin/bash

uv run fdp -c configs/ordering/order_analysis/full_ft.yaml
uv run fdp -c configs/ordering/order_analysis/lr_scheduler.yaml