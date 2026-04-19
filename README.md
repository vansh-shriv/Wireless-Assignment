"""
Drone Base Station (Drone-BS) 3D Placement in Wireless Cellular Networks
Paper: "On the Number and 3D Placement of Drone Base Stations in
        Wireless Cellular Networks"
Authors: Elham Kalantari, Halim Yanikomeroglu, Abbas Yongacoglu (2016)

This script reproduces ALL results from the paper in order:
  1. Air-to-ground path loss model  → Fig. 2
  2. Scenario I  (non-uniform density)  → Fig. 3a, 3b, Fig. 4, Fig. 5
  3. Scenario II (Gaussian + uniform)   → Fig. 6a, 6b, Fig. 7, Fig. 8

Dependencies: numpy, matplotlib, scipy
pip install requirements.txt

Run : python drone_bs_placement.py

All plots will be saved in Results Folder
"""