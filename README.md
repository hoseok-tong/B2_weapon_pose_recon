### Overview
Vedo를 사용, camera 위치 and Aruco marker pose 2d -> marker pose 3d 로 visualize code
camera parameter 
- MVS .txt
- metashape .xml
- matlab .mat
각각 case 마다 intrinsic, extrinsic parsing 하는 방법만 다름
Vedo camera visualize 코드는 고정됨

marker pose 2d 는 lib-dt-apriltags ( https://github.com/duckietown/lib-dt-apriltags ) code 로 추출함
