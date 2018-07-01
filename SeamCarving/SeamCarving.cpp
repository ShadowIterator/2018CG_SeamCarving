#define _CRT_SECURE_NO_WARNINGS
#include<cstdio>
#include<algorithm>
#include<cstdlib>
#include<ctime>
#include<cstring>
#include<iostream>
#include<string>
#include<vector>
#include<cmath>

#include <iostream>  
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  

using namespace std;
using namespace cv;

const int INF = 1 << 27;
const int MAXN = 2450;
int n, m;
int N, M;
int Dx[MAXN][MAXN];
int Dy[MAXN][MAXN];
int E[MAXN][MAXN];

int ifx[MAXN][MAXN];	//from up to down
int fx[MAXN][MAXN];
int ify[MAXN][MAXN];	//from left to right
int fy[MAXN][MAXN];
int seamx[MAXN];
int seamy[MAXN];
int nx[MAXN][MAXN];
int ny[MAXN][MAXN];
int Ny[MAXN][MAXN];
int Oy[MAXN][MAXN];

int elgseamx[MAXN][MAXN];
int elgtot;
bool mark[MAXN][MAXN];
char weight[MAXN][MAXN];

inline int Abs(int x)
{
	return x > 0 ? x : -x;
}

inline void calcEngy(Mat* img)
{
	for (int i = 1; i<n - 1; ++i)
		for (int j = 1; j < m - 1; ++j)
		{
			Dx[i][j] = (*img).at<uchar>(i,j + 1) - (*img).at<uchar>(i,j - 1);
			Dy[i][j] = (*img).at<uchar>(i + 1,j) - (*img).at<uchar>(i - 1,j);
			E[i][j] = Abs(Dx[i][j]) + Abs(Dy[i][j]);
			if (weight[i][j] == 1)
				E[i][j] = INF >> 10;
			else if (weight[i][j] == -1)
				E[i][j] = -(INF >> 10);
		}
	E[MAXN - 1][MAXN - 1] = INF;
}

inline bool oor(int x, int y)
{
	return x <= 0 || x >= n-1 || y < 1 || y >= m-1;
}

inline void minArg(int f[][MAXN], int x1, int y1, int x2, int y2, int x3, int y3, int& ansx, int& ansy)
{
	if (oor(x1, y1)) x1 = y1 = MAXN - 1;
	if (oor(x2, y2)) x2 = y2 = MAXN - 1;
	if (oor(x3, y3)) x3 = y3 = MAXN - 1;

	ansx = x1; ansy = y1;
	if (f[x2][y2] < f[ansx][ansy])
	{
		ansx = x2;
		ansy = y2;
	}
	if (f[x3][y3] < f[ansx][ansy])
	{
		ansx = x3;
		ansy = y3;
	}

}

inline void DPx()	//calc fx
{
	fx[MAXN - 1][MAXN - 1] = INF;
	int tx, ty;
	for (int j = 0; j < m; ++j)
		ifx[1][j] = -1, fx[1][j] = E[1][j];
	for (int i = 2; i < n-1; ++i)
		for (int j = 1; j < m-1; ++j)
		{
			minArg(fx, i - 1, j - 1, i - 1, j, i - 1, j + 1, tx, ty);
			fx[i][j] = E[i][j] + fx[tx][ty];
			ifx[i][j] = ty;
		}
	int miny = 1;
	for (int j = 1; j < m - 1; ++j)
	{
		if (fx[n - 2][miny] > fx[n - 2][j]) miny = j;
	}
	seamx[n - 1] = miny;
	for (int i = n - 2; i >= 1; --i)
	{
		seamx[i] = miny;
		miny = ifx[i][miny];
	}
	seamx[0] = seamx[1];
}

inline void DPy()	//calc fy
{
	fy[MAXN - 1][MAXN - 1] = INF;
	int tx, ty;
	for (int i = 0; i < n; ++i)
		ify[i][1] = -1, fy[i][1] = E[i][1];
	for (int j = 2; j < m - 1; ++j)
		for (int i = 1; i < n - 1; ++i)
		{
			minArg(fy, i - 1, j - 1, i, j - 1, i + 1, j - 1, tx, ty);
			fy[i][j] = E[i][j] + fy[tx][ty];
			ify[i][j] = tx;
		}
	int minx = 1;
	for (int i = 1; i < n-1; ++i)
		if (fy[i][m - 2] < fy[minx][m - 2]) minx = i;
	for (int j = m - 2; j >= 1; --j)
	{
		seamy[j] = minx;
		minx = ify[minx][j];
	}
	seamy[m - 1] = seamy[m - 2];
	seamy[0] = seamy[1];
}

inline void cutx(Mat* img, Mat* img2)	//erase a col
{
	int cy;
	for (int i = 0; i < n; ++i)
	{
		cy = seamx[i];
		mark[nx[i][cy]][ny[i][cy]] = true;
		for (int j = cy; j + 1 < m; ++j)
		{
			(*img).at<uchar>(i, j) = (*img).at<uchar>(i, j + 1);
			(*img2).at<Vec3b>(i, j) = (*img2).at<Vec3b>(i, j + 1);
			nx[i][j] = nx[i][j + 1];
			ny[i][j] = ny[i][j + 1];
			weight[i][j] = weight[i][j + 1];
		}
	}
	--m;
}

inline void cuty(Mat* img, Mat* img2) //erase a row
{
	int cx;
	for (int j = 0; j < m; ++j)
	{
		cx = seamy[j];
		mark[nx[cx][j]][ny[cx][j]] = true;
		for (int i = cx; i + 1 < n; ++i)
		{
			(*img).at<uchar>(i,j) = (*img).at<uchar>(i+1,j);
			(*img2).at<Vec3b>(i,j) = (*img2).at<Vec3b>(i+1,j);
			ny[i][j] = ny[i + 1][j];
			nx[i][j] = nx[i + 1][j];
			weight[i][j] = weight[i + 1][j];
		}
	}
	--n;
}

inline void fillpoint(Mat* img,int i, int j)
{
	for (int x = i - 5; x < i + 5; ++x)
		for (int y = j - 5; y < j + 5; ++y)
			(*img).at<Vec3b>(x, y) = 0;
}

inline void elarge(Mat* img, int* seamx)
{
	++M;
	int cy;
	for (int i = 0; i < N; ++i)
	{
		cy = Ny[i][seamx[i]];
		for (int j = M - 1; j > cy; --j)
		{
			(*img).at<Vec3b>(i, j) = (*img).at<Vec3b>(i, j - 1);
			Oy[i][j] = Oy[i][j - 1];
			Ny[i][Oy[i][j]] = j;
		}
		if (cy == 0)
			(*img).at<Vec3b>(i, cy) = (*img).at<Vec3b>(i, cy + 1);
		else
		{
			uchar* bs = &(*img).at<uchar>(i, 3*cy);
			uchar* bs1= &(*img).at<uchar>(i, 3*cy-3);
			uchar* bs2 = &(*img).at<uchar>(i, 3*cy + 3);
			bs[0] = (bs1[0] + bs2[0]) / 2;
			bs[1] = (bs1[1] + bs2[1]) / 2;
			bs[2] = (bs1[2] + bs2[2]) / 2;

		}
			//(*img).at<Vec3b>(i, cy) = ((*img).at<Vec3b>(i, cy + 1) + (*img).at<Vec3b>(i, cy - 1)) / 2;
		
	}
	
}

/*
input:
	- 双向缩小 1.jpg 0
	- 横向拉伸 1.jpg 2
	- 对象保护 1.jpg 1 663 683 80 520
	- 对象移除 1.jpg -1 644 91 77 45
*/

int main(int argc, char** argv)
{
	string name;
	cin >> name;
	int x1, y1, x2, y2, op;
	cin >> op;
	cout << name << endl;
	if (op == 1)
	{
		cin >> x1 >> y1 >> x2 >> y2;
		for (int i = x1; i < x1 + x2; ++i)
			for (int j = y1; j < y1 + y2; ++j)
				weight[i][j] = 1;
	}
	else if (op == -1)
	{
		cin >> x1 >> y1 >> x2 >> y2;
		for (int i = x1; i < x1 + x2; ++i)
			for (int j = y1; j < y1 + y2; ++j)
				weight[i][j] = -1;
	}

	string of1, of2;
	string dir = "";
	string suffix;
	if (op == 0) suffix = "-cut.bmp";
	else if (op == 1) suffix = "-cut-protect.bmp";
	else if (op == -1) suffix = "-cut-remove.bmp";
	else suffix = "-enlarge.bmp";
	of1 = dir + name + suffix;
	of2 = dir + name + "ori" + suffix;
	Mat img = imread(name.c_str(), IMREAD_COLOR);
	Mat ori;
	Mat tori;
	img.copyTo(ori);
	img.copyTo(tori);

	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);

	n = img.rows;
	m = img.cols;

	for (int i = 0; i < n; ++i)
		for (int j = 0; j < m; ++j)
			nx[i][j] = i, ny[i][j] = j, Ny[i][j] = j, Oy[i][j] = j;

	int CY = n * 0.2;
	int CX = m * 0.2;


	for (int k = 0; k < CX; ++k)
	{
		cout << "cutting x " << k << " / " << CX << endl;
		calcEngy(&gray);
		DPx();
		if (op == 2)
		{
			for (int i = 0; i < n; ++i)
			{
				elgseamx[elgtot][i] = ny[i][seamx[i]];
			}
			++elgtot;
		}
		cutx(&gray, &img);
	}

	if (op != 2)
	{
		for (int k = 0; k < CY; ++k)
		{
			cout << "cutting y " << k << " / " << CY << endl;
			calcEngy(&gray);
			DPy();

			cuty(&gray, &img);
		}
	}

	if (op != 2)
	{
		for (int i = 0; i < ori.rows; ++i)
			for (int j = 0; j < ori.cols; ++j)
			{
				if (mark[i][j])
				{
					ori.at<Vec3b>(i, j)[1] = 255;
					ori.at<Vec3b>(i, j)[0] = ori.at<Vec3b>(i, j)[2] = 0;
				}
			}
		imwrite(of2, ori);
		cout << "imwrite-ori.done" << endl;

		cout << img.rows << " " << img.cols << endl;
		cout << n << " " << m << endl;
		Mat output = img(Range(0, n), Range(0, m));
		imwrite(of1, output);

		cout << "outputdone" << endl;
	}
	else
	{
		Mat dst(tori.rows, tori.cols + CX, CV_8UC3, Scalar(0, 0, 0));
		Mat dori(tori.rows, tori.cols + CX, CV_8UC3, Scalar(0, 0, 0));
		N = tori.rows;
		M = tori.cols;
		for (int i = 0; i < N; ++i)
			for (int j = 0; j < M; ++j)
				dst.at<Vec3b>(i, j) = tori.at<Vec3b>(i, j);
		for (int k = 0; k < elgtot; ++k)
			elarge(&dst,elgseamx[k]);
		imwrite(dir+name+"-enlarge.bmp", dst);
		dst.copyTo(dori);
		for (int i = 0; i<ori.rows; ++i)
			for (int j = 0; j < ori.cols; ++j)
			{
				if (mark[i][j])
				{
					dori.at<Vec3b>(i, Ny[i][j])[1] = 255;
					dori.at<Vec3b>(i, Ny[i][j])[0] = dori.at<Vec3b>(i, Ny[i][j])[2] = 0;
				}
			}
		imwrite(dir + name + "-ori-enlarge.bmp", dori);
	}
	return 0;	
}