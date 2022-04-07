#include <iostream>
#include <string>
#include <Windows.h>
#include <cmath>

#include <eigen3\Eigen\Dense>
#include <eigen3/Eigen/SVD>
#include <eigen3/Eigen/Core>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vtkSmartPointer.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolyData.h>
#include <vtkPLYWriter.h>
#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkImageViewer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleImage.h>
#include <vtkLight.h>
#include <vtkLightCollection.h>
#include <vtkRenderer.h>
#include <vtkCellArray.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkFloatArray.h>
#include <vtkTriangle.h>
#include "vtkAutoInit.h"

VTK_MODULE_INIT(vtkRenderingOpenGL2)
VTK_MODULE_INIT(vtkInteractionStyle)

//https://eigen.tuxfamily.org/index.php?title=Main_Page
//https://www.programmersought.com/article/84351161423/
using namespace std;
using namespace Eigen;

#pragma warning(disable : 4996)

const int NUM_IMG = 23;
const int NUM_PIXEL_X = 2048;
const int NUM_PIXEL_Y = 2048;

RECT GetBoundingBox(const unsigned char* mask);
void LoadFile(const string& str, unsigned char* buffer);
unsigned char*  GetLightDirFromSphere(RECT bb, const unsigned char* image, float* output);
cv::Mat globalHeights(float* Pgrads, float* Qgrads, int rows, int cols);
Eigen::MatrixXd pinv_eigen_based(Eigen::MatrixXd & origin);
void displayMesh(int width, int height, cv::Mat Z);
int main()
{

	//MatrixXd a(30, 30);
	//a(0, 0) = 1;
	//a.resize(10, 10);
	//a(7, 7) = 7;
	//Matrix<unsigned char, NUM_PIXEL_X, NUM_PIXEL_Y> mask;
	//Matrix<Matrix<unsigned char, NUM_PIXEL_X, NUM_PIXEL_Y>, 1, 12> CALL;
	//CALL[1](10, 10) = 10;
	//mask(10, 10) = 10;
	//mask(10, 10) = 20;

	string CALIBRATION = "D:\\Projects\\EditPhotometricStereo\\Debug\\image\\chrome4\\chrome4.";
	string MODEL = "D:\\Projects\\EditPhotometricStereo\\Debug\\image\\coin\\coin.";

	unsigned char** calibImages = new unsigned char*[NUM_IMG];
	unsigned char** modelImages = new unsigned char*[NUM_IMG];
	float Lights[NUM_IMG][3];

	unsigned char* Mask = new unsigned char[NUM_PIXEL_X * NUM_PIXEL_Y];
	unsigned char* ModelMask = new unsigned char[NUM_PIXEL_X * NUM_PIXEL_Y];


	//마스크 이미지 파일 로딩
	LoadFile(CALIBRATION + "mask.bmp", Mask);
	LoadFile(MODEL + "mask.bmp", ModelMask);

	//ROI설정
	RECT bb = GetBoundingBox(Mask);
	int xSize = bb.right - bb.left;
	int ySize = bb.bottom - bb.top;

	RECT bb2 = GetBoundingBox(ModelMask);
	int modelxSize = bb2.right - bb2.left;
	int modelySize = bb2.bottom - bb2.top;

	//실물 이미지 로딩
	for (int i = 0; i < NUM_IMG; ++i)
	{
		string str = to_string(i);
		unsigned char* cali = new unsigned char[NUM_PIXEL_X * NUM_PIXEL_Y];
		LoadFile(CALIBRATION + str + ".bmp", cali);
		calibImages[i] = cali;
		unsigned char* img = new unsigned char[NUM_PIXEL_X * NUM_PIXEL_Y];
		LoadFile(MODEL + str + ".bmp", img);
		modelImages[i] = img;

		unsigned char* tempROIedModel = new unsigned char[modelxSize * modelySize];

		////임시코드
		//cv::Mat model(NUM_PIXEL_X, NUM_PIXEL_Y, CV_32F, cv::Scalar::all(0));
		//for (int y = 0; y < NUM_PIXEL_Y; ++y)
		//{
		//	for (int x = 0; x < NUM_PIXEL_X; ++x)
		//	{
		//		model.at<float>(cv::Point(x, y)) = modelImages[i][y * NUM_PIXEL_X + x];
		//	}
		//}
		//string tempstr = "modelTemp";
		//char num[20];
		//itoa(i, num, 10);
		//tempstr += num;
		//tempstr += ".png";
		//cv::imwrite("D:\\Projects\\EditPhotometricStereo\\Debug\\" + tempstr, model);


		//ROI만큼만 저장하기
		for (int y = 0; y < modelySize; ++y)
		{
			for (int x = 0; x < modelxSize; ++x)
			{
				long imgIdx = bb2.top *  NUM_PIXEL_X + bb2.left + (y * NUM_PIXEL_X) + x;
				tempROIedModel[y * modelxSize + x] = modelImages[i][imgIdx];
			}
		}

		float* fCalib = new float[3];

		//광원방향xyz 구하기
		GetLightDirFromSphere(bb, cali, fCalib);
		Lights[i][0] = fCalib[0];
		Lights[i][1] = fCalib[1];
		Lights[i][2] = fCalib[2];

		memset(modelImages[i], 0, NUM_PIXEL_X *NUM_PIXEL_Y);

		for (int y = 0; y < modelySize; ++y)
		{
			for (int x = 0; x < modelxSize; ++x)
			{
				modelImages[i][bb2.top *  NUM_PIXEL_X + bb2.left + y * NUM_PIXEL_X + x] = tempROIedModel[y * modelxSize + x];
			}
		}
		delete(fCalib);
	}

	MatrixXd LightsMat(NUM_IMG, 3);
	for (int i = 0; i < NUM_IMG; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			LightsMat(i, j) = Lights[i][j];
		}
	}

	const int height = NUM_PIXEL_Y;
	const int width = NUM_PIXEL_X;

	MatrixXd LightsInvMat(3, NUM_IMG);
	double LightsInv[3][NUM_IMG];
	LightsInvMat = pinv_eigen_based(LightsMat);
	
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < NUM_IMG; ++j)
		{
			LightsInv[i][j] = LightsInvMat(i, j);
		}
	}

	//임시코드

	float fBuffer[NUM_IMG][3] = {};

	for (int i = 0; i < NUM_IMG; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			fBuffer[i][j] = LightsInv[j][i];
		}
	}

	char strbuffer[5] = ", ";
	char strbuffer2[5] = "\n";
	char tempstr[NUM_IMG][100];
	string filestr;
	for (int i = 0; i < NUM_IMG; ++i)
	{
		sprintf(tempstr[i], "%f, %f, %f\n", fBuffer[i][0], fBuffer[i][1], fBuffer[i][2]);
		filestr += tempstr[i];
	}
	FILE* f = fopen("D:\\Projects\\EditPhotometricStereo\\Debug\\light.txt", "w");
	fputs(filestr.c_str(), f);
	fclose(f);

	//char tempstr[NUM_IMG  * 3][100];
	//FILE* f = fopen("C:\\Users\\DeeDiim\\source\\repos\\ConsoleApplicationINV\\light.txt", "r");
	//for(int i = 0; i < NUM_IMG * 3; ++i)
	//	fgets(tempstr[i], 100, f);
	//
	//for (int i = 0; i < 3; ++i)
	//{
	//	for (int j = 0; j < NUM_IMG; ++j)
	//	{
	//
	//		LightsInv[i][j] = atof(tempstr[i * NUM_IMG + j]);
	//	}
	//}
	//
	//fclose(f);

	
	float l[NUM_IMG];

	float * Normals = new float[height * width * 3];
	memset(Normals, 0, sizeof(float) * height * width * 3);
	float * Pgrads = new float[height * width];
	memset(Pgrads, 0, sizeof(float) * height * width);
	float * Qgrads = new float[height * width];
	memset(Qgrads, 0, sizeof(float) * height * width);

	for (int x = 0; x < width; ++x)
	{
		for (int y = 0; y < height; ++y)
		{
			for (int i = 0; i < NUM_IMG; ++i)
			{
				l[i] = modelImages[i][x * height + y];
			}
			Vector3f n(0,0,0);
			for (int i = 0; i < 3; ++i)
			{
				double sum = 0;
				for (int j = 0; j < NUM_IMG; ++j)
				{
					sum += l[j] * LightsInv[i][j];
				}
				n(i) = sum;
			}
			float p = sqrt(n.dot(n));
			if(p > 0) { n = n / p; }
			if (n(2, 0) == 0) { n(2, 0) = 1.0; }
			int legit = 1;

			for (int i = 0; i < NUM_IMG; i++) {
				legit *= modelImages[i][x * height + y] >= 0;
			}

			if (legit) {

				if (n(0, 0) / n(2, 0) != 0 || n(1, 0) / n(2, 0) != 0)
				{
					if (ModelMask[x* height + y] != 0)
					{
						Normals[y * width + x * 3] = n(0, 0);
						Normals[y * width + x * 3 + 1] = n(1, 0);
						Normals[y * width + x * 3 + 2] = n(2, 0);
						Pgrads[y * width + x] = n(0, 0) / n(2, 0);
						Qgrads[y * width + x] = n(1, 0) / n(2, 0);
					}
				}
			}
			else
			{
				float nullvec[3] = { 0.f,0.f,1.0f };
				Normals[y * width + x * 3] = nullvec[0];
				Normals[y * width + x * 3 + 1] = nullvec[1];
				Normals[y * width + x * 3 + 2] = nullvec[2];

				Pgrads[y * width + x] = 0.0f;
				Qgrads[y * width + x] = 0.0f;
			}
		}
	}

	cv::Mat INH(height, width, 0, cv::Scalar::all(0));
	cv::Mat INV(height, width, 0, cv::Scalar::all(0));

	for (int x = 0; x < width; ++x)
	{
		for (int y = 0; y < height; ++y)
		{
			float pixelValueP = Pgrads[x*height + y];
			float pixelValueQ = Qgrads[x*height + y];

			pixelValueP += 1.f;
			pixelValueQ += 1.f;
			pixelValueP *= 127.f;
			pixelValueQ *= 127.f;

			if (pixelValueP > 255)
				pixelValueP = 255;
			if (pixelValueP < 0)
				pixelValueP = 0;

			if (pixelValueQ > 255)
				pixelValueQ = 255;
			if (pixelValueQ < 0)
				pixelValueQ = 0;

			INH.at<uchar>(cv::Point(x, y)) = (unsigned char)pixelValueP;
			INV.at<uchar>(cv::Point(x, y)) = (unsigned char)pixelValueQ;
		}
	}

	cv::imwrite("D:\\Projects\\EditPhotometricStereo\\Debug\\Pgrade2.png", INH);
	cv::imwrite("D:\\Projects\\EditPhotometricStereo\\Debug\\Qgrade2.png", INV);


	cv::Mat Z = globalHeights(Pgrads, Qgrads, width, height);

	displayMesh(height, width, Z);

	//딜리트 작업
	delete(Mask);
	delete(ModelMask);

	delete(Normals);
	delete(Pgrads);
	delete(Qgrads);

	for (int i = 0; i < NUM_IMG; ++i)
	{
		delete(calibImages[i]);
		delete(modelImages[i]);
	}
}

RECT GetBoundingBox(const unsigned char* mask)
{
	//엣지 따기 노이즈가 없는 마스크맵을 전제로
	RECT rt{};
	bool breakloop = false;
	for (int y = 0; y < NUM_PIXEL_Y; ++y)
	{
		for (int x = 0; x < NUM_PIXEL_X; ++x)
		{
			if (mask[y * NUM_PIXEL_X + x] == 255)
			{
				rt.top = y;
				breakloop = true;
				break;
			}
		}
		if (breakloop)
			break;
	}

	breakloop = false;

	for (int x = 0; x < NUM_PIXEL_X; ++x)
	{
		for (int y = 0; y < NUM_PIXEL_Y; ++y)
		{
			if (mask[y * NUM_PIXEL_X + x] == 255)
			{
				rt.left = x;
				breakloop = true;
				break;
			}
		}
		if (breakloop)
			break;
	}

	breakloop = false;

	for (int y = NUM_PIXEL_Y; y > 0; --y)
	{
		for (int x = NUM_PIXEL_Y; x > 0; --x)
		{
			if (mask[y * NUM_PIXEL_X + x] == 255)
			{
				rt.bottom = y;
				breakloop = true;
				break;
			}
		}
		if (breakloop)
			break;
	}

	breakloop = false;

	for (int x = NUM_PIXEL_X; x > 0; --x)
	{
		for (int y = NUM_PIXEL_Y; y > 0; --y)
		{
			if (mask[y * NUM_PIXEL_X + x] == 255)
			{
				rt.right = x;
				breakloop = true;
				break;
			}
		}
		if (breakloop)
			break;
	}
	return rt;
}

void LoadFile(const string& str, unsigned char* buffer)
{
	FILE* f;
	f = fopen(str.c_str(), "rb");
	unsigned char byte[8] = {};
	//header
	BITMAPFILEHEADER hf;
	BITMAPINFOHEADER hInfo;
	fread(&hf, sizeof(BITMAPFILEHEADER), 1, f);
	//chk bmp
	if (hf.bfType == 0x4D42)
	{

		fread(&hInfo, sizeof(BITMAPINFOHEADER), 1, f);

		//grayscale 확인
		if (hInfo.biBitCount != 8)
			assert(nullptr);

		// BMP Pallete
		RGBQUAD hRGB[256];
		fread(hRGB, sizeof(RGBQUAD), 256, f);

		// Memory y상하가 뒤집혀서 저장되어있다.
		long otherSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + sizeof(RGBQUAD) * 256;
		long pixelSize = hf.bfSize - otherSize;
		unsigned char *lpImg = new unsigned char[pixelSize];
		fread(lpImg, sizeof(char), pixelSize, f);

		for (int i = 0; i < NUM_PIXEL_Y; ++i)
		{
			for (int j = 0; j < NUM_PIXEL_X; ++j)
			{
				long idx = (NUM_PIXEL_Y - i) * NUM_PIXEL_X + j;
				buffer[i * NUM_PIXEL_X + j] = lpImg[idx];
			}
		}
		fclose(f);
		delete(lpImg);
		return;
	}
}

unsigned char* GetLightDirFromSphere(RECT bb, const unsigned char* image, float* output)
{
	const float radius = (bb.right - bb.left) / 2.0f;
	int xSize = bb.right - bb.left;
	int ySize = bb.bottom - bb.top;
	unsigned char* Binary = new unsigned char[xSize * ySize];
	int* BinaryX = new int[xSize * ySize];
	int* BinaryY = new int[xSize * ySize];

	int nSize = 0;
	for (int y = 0; y < ySize; ++y)
	{
		for (int x = 0; x < xSize; ++x)
		{
			long index = bb.top *  NUM_PIXEL_X + bb.left + (y * NUM_PIXEL_X) + x;
			BinaryX[y* xSize + x] = 0;
			BinaryY[y* xSize + x] = 0;
			Binary[y* xSize + x] = image[index];
			if (image[index] == 255)
			{
				BinaryX[y* xSize + x] = x;
				BinaryY[y* xSize + x] = y;
				nSize++;
			}
		}
	}

	//센터 계산
	long long sumX = 0;
	long long sumY = 0;
	for (int y = 0; y < ySize; ++y)
	{
		for(int x = 0; x < xSize; ++x)
		{
			sumX += BinaryX[y * xSize + x];
			sumY += BinaryY[y * xSize + x];
		}
	}
	
	float cx = sumX / nSize;
	float cy = (sumY / nSize);

	float x = (cy - radius) / radius;
	float y = (cx - radius) / radius;
	float z = sqrt(1.0 - pow(x, 2.0) - pow(y, 2.0));

	output[0] = x;output[1] = y;output[2] = z;

	return Binary;
}

cv::Mat globalHeights(float* Pgrads, float* Qgrads, int rows, int cols)
{
	cv::Mat MatPgrads(rows, cols, CV_32F, cv::Scalar::all(0));
	cv::Mat MatQgrads(rows, cols, CV_32F, cv::Scalar::all(0));

	for (int x = 0; x < rows; ++x)
	{
		for (int y = 0; y < cols; ++y)
		{
			MatPgrads.at<float>(cv::Point(x, y)) = Pgrads[x  * cols + y];
			MatQgrads.at<float>(cv::Point(x, y)) = Qgrads[x  * cols + y];
		}
	}

	cv::Mat P(rows, cols, CV_32FC2, cv::Scalar::all(0));
	cv::Mat Q(rows, cols, CV_32FC2, cv::Scalar::all(0));
	cv::Mat Z(rows, cols, CV_32FC2, cv::Scalar::all(0));

	float lambda = 1.0f;
	float mu = 1.0f;

	cv::dft(MatPgrads, P, cv::DFT_COMPLEX_OUTPUT);
	cv::dft(MatQgrads, Q, cv::DFT_COMPLEX_OUTPUT);
	for (int i = 0; i < MatPgrads.rows; i++) {
		for (int j = 0; j < MatPgrads.cols; j++) {
			if (i != 0 || j != 0) {
				float u = sin((float)(i * 2 * CV_PI / rows));
				float v = sin((float)(j * 2 * CV_PI / cols));

				float uv = pow(u, 2) + pow(v, 2);
				float d = (1.0f + lambda)*uv + mu * pow(uv, 2);
				Z.at<cv::Vec2f>(i, j)[0] = (u*P.at<cv::Vec2f>(i, j)[1] + v * Q.at<cv::Vec2f>(i, j)[1]) / d;
				Z.at<cv::Vec2f>(i, j)[1] = (-u * P.at<cv::Vec2f>(i, j)[0] - v * Q.at<cv::Vec2f>(i, j)[0]) / d;
			}
		}
	}

	/* setting unknown average height to zero */
	Z.at<cv::Vec2f>(0, 0)[0] = 0.0f;
	Z.at<cv::Vec2f>(0, 0)[1] = 0.0f;

	cv::dft(Z, Z, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

	return Z;
}

//Using the Eigen library, using the SVD decomposition method to solve the matrix pseudo - inverse, the default error er is 0
Eigen::MatrixXd pinv_eigen_based(Eigen::MatrixXd & origin)
{
	// perform svd decomposition
	Eigen::JacobiSVD<Eigen::MatrixXd> svd_holder(origin,
	Eigen::ComputeThinU |
	Eigen::ComputeThinV);
	// Build SVD decomposition results
	Eigen::MatrixXd U = svd_holder.matrixU();
	Eigen::MatrixXd V = svd_holder.matrixV();
	Eigen::MatrixXd D = svd_holder.singularValues();

	// Build the S matrix
	Eigen::MatrixXd S(V.cols(), U.cols());
	S.setZero();

	for (unsigned int i = 0; i < D.size(); ++i) {

		if (D(i, 0) > 0) {
			S(i, i) = 1 / D(i, 0);
		}
		else {
			S(i, i) = 0;
		}
	}

	// pinv_matrix = V * S * U^T
	return V * S * U.transpose();
}

void displayMesh(int width, int height, cv::Mat Z)
{
	/* creating visualization pipeline which basically looks like this:
	 vtkPoints -> vtkPolyData -> vtkPolyDataMapper -> vtkActor -> vtkRenderer */
	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
	vtkSmartPointer<vtkPolyDataMapper> modelMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	vtkSmartPointer<vtkActor> modelActor = vtkSmartPointer<vtkActor>::New();
	vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
	vtkSmartPointer<vtkCellArray> vtkTriangles = vtkSmartPointer<vtkCellArray>::New();

	/* insert x,y,z coords */
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			points->InsertNextPoint(x, y, Z.at<float>(y, x));
		}
	}

	/* setup the connectivity between grid points */
	vtkSmartPointer<vtkTriangle> triangle = vtkSmartPointer<vtkTriangle>::New();
	triangle->GetPointIds()->SetNumberOfIds(3);
	for (int i = 0; i < height - 1; i++) {
		for (int j = 0; j < width - 1; j++) {
			triangle->GetPointIds()->SetId(0, j + (i*width));
			triangle->GetPointIds()->SetId(1, (i + 1)*width + j);
			triangle->GetPointIds()->SetId(2, j + (i*width) + 1);
			vtkTriangles->InsertNextCell(triangle);
			triangle->GetPointIds()->SetId(0, (i + 1)*width + j);
			triangle->GetPointIds()->SetId(1, (i + 1)*width + j + 1);
			triangle->GetPointIds()->SetId(2, j + (i*width) + 1);
			vtkTriangles->InsertNextCell(triangle);
		}
	}
	polyData->SetPoints(points);
	polyData->SetPolys(vtkTriangles);

	/* create two lights */
	vtkSmartPointer<vtkLight> light1 = vtkSmartPointer<vtkLight>::New();
	light1->SetPosition(-1, 1, 1);
	renderer->AddLight(light1);
	vtkSmartPointer<vtkLight> light2 = vtkSmartPointer<vtkLight>::New();
	light2->SetPosition(1, -1, -1);
	renderer->AddLight(light2);

	/* meshlab-ish background */
	modelMapper->SetInputData(polyData);
	renderer->SetBackground(.45, .45, .9);
	renderer->SetBackground2(.0, .0, .0);
	renderer->GradientBackgroundOn();
	vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
	renderWindow->AddRenderer(renderer);
	modelActor->SetMapper(modelMapper);

	/* setting some properties to make it look just right */
	modelActor->GetProperty()->SetSpecularColor(1, 1, 1);
	modelActor->GetProperty()->SetAmbient(0.2);
	modelActor->GetProperty()->SetDiffuse(0.2);
	modelActor->GetProperty()->SetInterpolationToPhong();
	modelActor->GetProperty()->SetSpecular(0.8);
	modelActor->GetProperty()->SetSpecularPower(8.0);

	renderer->AddActor(modelActor);
	vtkSmartPointer<vtkRenderWindowInteractor> interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
	interactor->SetRenderWindow(renderWindow);

	/* export mesh */
	vtkSmartPointer<vtkPLYWriter> plyExporter = vtkSmartPointer<vtkPLYWriter>::New();
	plyExporter->SetInputData(polyData);
	plyExporter->SetFileName("export.ply");
	plyExporter->SetColorModeToDefault();
	plyExporter->SetArrayName("Colors");
	plyExporter->Update();
	plyExporter->Write();

	/* render mesh */
	renderWindow->Render();
	interactor->Start();
}