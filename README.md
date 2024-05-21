### Overview
Vedo를 사용, camera 위치 and Aruco marker pose 2d -> marker pose 3d 로 visualize code
camera parameter 
- MVS .txt
- metashape .xml
- matlab .mat
각각 case 마다 intrinsic, extrinsic parsing 하는 방법만 다름
Vedo camera visualize 코드는 고정됨

marker pose 2d 는 lib-dt-apriltags ( https://github.com/duckietown/lib-dt-apriltags ) code 로 추출함


### TODO
- 추정한 marker pose3d 데이터를 통해 총구 방향 RT 추출
- 1. 추정한 marker psoe3d 를 정육면체 constraint 를 통해 튀는 데이터, occlusion 을 보정해 추출하는 방법
  2. 추정한 marker pose3d 에 정육면체 mesh 를 icp 함으로 총구 방향 RT 를 추출하는 방법
- 정밀한 정육면체 큐브를 이용한 데이터 재취득 필요
- Opti-Track을 이용해 추출한 3d pose GT 와 quality 비교
