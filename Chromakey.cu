#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

#define CHECK(value) {                                          \
    cudaError_t _m_cudaStat = value;                                        \
    if (_m_cudaStat != cudaSuccess) {                                       \
        cout<< "Error:" << cudaGetErrorString(_m_cudaStat) \
            << " at line " << __LINE__ << " in file " << __FILE__ << "\n"; \
        exit(1);                                                            \
    } }

#define TILE_WIDTH 8

__global__ void setBackColor(uchar3 * pxs, uchar3* back, int length, int minHue, int maxHue)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    if (i < length)
    {
        uchar *p = (uchar*)(&pxs[i]);
        uchar *bp = (uchar*)(&back[i]);

        double r = p[2] / 255.0, g = p[1] / 255.0, b = p[0] / 255.0;
        double cmax = r > g ? r : g;
        cmax = cmax > b ? cmax : b;

        double cmin = r < g ? r : g;
        cmin = cmin < b ? cmin : b;

        double delta = cmax - cmin;

        double h;

        if (delta < 0.0001 && delta > -0.0001)
          h = 0.0;
        else if (cmax == r)
        {
            h = 60 * ((int)((g - b) / delta) % 6);
        }
        else if (cmax == g)
        {
            h = 60 * (((b - r) / delta) + 2);
        }
        else
        {
            h = 60 * (((r - g) / delta) + 4);
        }

        if (h >= minHue && h < maxHue)
        {
            p[0] = bp[0];
            p[1] = bp[1];
            p[2] = bp[2];
        }
    }
}

int main(int argc, char* argv[])
{
    VideoCapture cap(argv[0]);
    VideoCapture backCap(argv[1]);

    if(!cap.isOpened() || !backCap.isOpened()) return 0;

    Mat frame, back;
    VideoWriter outputVideo;

    int minHue, maxHue;
    sscanf(argv[3], "%d", &minHue);
    sscanf(argv[4], "%d", &maxHue);

    int width = 640;
    int height = 360;
    int fps = 24;

    namedWindow("1");
    namedWindow("2");

    Size winSize(width, height);
    outputVideo.open(argv[2], CV_FOURCC('M','J','P','G'), fps, winSize, true);

    if(!outputVideo.isOpened()) return 0;

    uchar3 *devFrame, *devBack;

    CHECK(  cudaMalloc(&devFrame, 3 * width * height)  );
    CHECK(  cudaMalloc(&devBack, 3 * width * height)  );

    while(true)
    {
        if (!cap.read(frame)) break;

        int N = frame.rows * frame.cols;
        uchar3 *frameData = (uchar3*)frame.data;
        uchar3 *backData;
        if(backCap.read(back))
          backData = (uchar3*)back.data;
        else
        {
            backData = frameData;
            cout << "back";
        }

        CHECK( cudaMemcpy(devFrame, frameData, 3* N,cudaMemcpyHostToDevice) );
        CHECK( cudaMemcpy(devBack, backData, 3* N,cudaMemcpyHostToDevice) );

        imshow("1", frame);

        setBackColor<<<(N + TILE_WIDTH - 1) / TILE_WIDTH, TILE_WIDTH>>>(devFrame, devBack, N, minHue, maxHue);
        CHECK(  cudaMemcpy(frameData, devFrame, 3*N,cudaMemcpyDeviceToHost)  );
        CHECK(  cudaGetLastError()  );
        imshow("2", frame);

        outputVideo.write(frame);
        waitKey(20);
    }

    cudaFree(&devFrame);
    cudaFree(&devBack);

    return 0;
}
