# North Polar Project

## Project Description

The North Polar Project aims to design an efficient spatial coverage algorithm for urban areas in the Arctic region, targeting dense building zones with minimal geographic footprint while avoiding the collection of redundant data (e.g., forests, rivers, and other non-urban surfaces). The project leverages the powerful geospatial analytics capabilities of Google Earth Engine (GEE) and the interactive development environment of Google Colab to enable scalable and high-performance remote sensing processing.

This system can be extended to enable real-time monitoring of buildings in the Arctic region, with automated dataset generation. It has the potential to advance the current capabilities of remote sensing satellites in Arctic applications.

The project also provides a comprehensive suite of code interfaces and tools, including multi-threaded downloading of Sentinel-2 satellite imagery from GEE, a complete annotation system (to be open-sourced soon), a customized project map based on GMap, model inference interfaces, and standardized training datasets. These resources are designed to lower technical barriers and empower researchers and industry practitioners to quickly deploy, adapt, and extend the system—advancing automated and intelligent urban monitoring in the Arctic region.

## Project Background
- Geographic Scope: The project covers the entire Arctic region, including countries such as Finland, Iceland, Russia, and Canada.
- Challenge: Acquiring high-resolution geospatial data directly is costly and often includes large amounts of data unrelated to buildings.
- Objective: Develop an algorithm to cover all essential urban (building-dense) areas with minimal geographic footprint.

## Technical Requirements
- Geospatial processing based on the Google Earth Engine (GEE) platform.
- Code development and execution using Google Colab.

## Progress
2025-08-05: First Draft Version completed 🎉. Both Windows and Mac system can run this system. Feel free to download. Please run `north_polar/main.ipynb`

## Future
2025-08-19: We have decided to further develop a more precise semantic segmentation model, where each training image will contain two labels: background areas labeled as 0 and urban areas labeled as 1.

2025-08-25: Third version of annotation tool is developing, with automated dataset cleaning pipeline.
