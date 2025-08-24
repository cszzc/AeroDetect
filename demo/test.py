import sys
sys.path.append("/root/autodl-tmp/ultralytics-mainPro")

from ultralytics.nn.addmodules import ADDWConvHead

if __name__ == "__main__":
    model = ADDWConvHead(nc=80, ch=[256, 512, 1024])
    print("111", model)