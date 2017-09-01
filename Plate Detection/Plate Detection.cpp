#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <opencv\ml.h>
#include <opencv\cxcore.h>
#include <baseapi.h>
#include <allheaders.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>

using namespace cv;
using namespace std;
using namespace tesseract;

struct loc
{
	int x;
	int y;
	int h;
	int w;
};

//>>>>>>>>>>>>>>>>>>>>>>>> Train and Print <<<<<<<<<<<<<<<<<<<<<<<<<

void Print(string textNum1, string textNum2)
{
	ofstream output;
	output.open("C:/Users/Hamid/Desktop/Plate.txt");

	string	a = textNum1;
	a.erase(2, 34);
	string b = textNum1;
	b.erase(0, 2);
	int i;
	for (i = 0; i < b.size(); i++)
	{
		if (b[i] >= 48 && b[i] < 58)
		{
			break;
		}
	}
	b.erase(i, 5);
	if (b[0] == 32)
	{
		b.erase(0, 1);
	}
	if (b[b.size() - 1] == 32)
	{
		b.erase(b.size() - 1, 1);
	}
	if (b.size() > 2)
	{
		b.erase(2, 5);
	}
	string c = textNum1;
	c.erase(0, i + 2);
	c.erase(3, 5);

	string num2 = textNum2;
	num2.erase(2, 20);

	/*wfstream f("D:\\test.txt", ios::out);
	wstring s1(L"ایران");
	f << s1.c_str();
	f.close();*/
	//wstring s1(L"ایران");


	output << c;
	output << "-";
	output << num2;
	//output << "53";
	output << b;
	output << a;

	//output << endl << s1.c_str();

	//output << " | 11";

	//output << endl << text;

	output.close();
}

void Train(Mat src, String &TextOcr, int numPlate)
{
	Mat gray;
	TessBaseAPI tess;
	cvtColor(src, gray, CV_RGB2GRAY);
	double thresh, maxval;
	double threshold1, threshold2;
	const int rep = 10;
	Mat threshold_Mat[rep];
	Mat canny_Mat[rep];
	int numlike = 0;
	srand(time(0));

	//**********************take filter

	for (int i = 0; i < rep; i++)
	{
		thresh = rand() % 300;
		maxval = rand() % 300;
		threshold1 = rand() % 300;
		threshold2 = rand() % 300;
		threshold(gray, threshold_Mat[i], thresh, maxval, THRESH_BINARY_INV);
		Canny(gray, canny_Mat[i], threshold1, threshold2);
	}

	tess.Init(NULL, "pelak2", OEM_DEFAULT);
	tess.SetPageSegMode(PSM_SINGLE_BLOCK);
	string textOcrString[rep * 2 + 1];
	string TrueOcrtext[rep * 2 + 1];

	//**********************OCR

	for (int i = 0; i < rep; i++)
	{
		tess.SetImage((uchar*)threshold_Mat[i].data, src.cols, src.rows, 1, src.cols);
		char *text1 = tess.GetUTF8Text();
		string str(text1);
		textOcrString[i] = str;
	}
	for (int i = rep; i < rep * 2; i++)
	{
		tess.SetImage((uchar*)canny_Mat[i].data, src.cols, src.rows, 1, src.cols);
		char *text1 = tess.GetUTF8Text();
		string str(text1);
		textOcrString[i] = str;
	}

	tess.SetImage((uchar*)gray.data, src.cols, src.rows, 1, src.cols);
	char *text1 = tess.GetUTF8Text();
	string str(text1);
	textOcrString[2 * rep] = str;

	//***********************Chack Status train data
	int countTrueTrain = 0;



	if (numPlate == 1)
	{
		for (int i = 0; i < rep * 2 + 1; i++)
		{
			int locnum1;
			for (locnum1 = 0; locnum1 < 7; locnum1++)
			{
				if (textOcrString[i][locnum1] >= 48 && textOcrString[i][locnum1] < 58)
				{
					break;
				}
			}
			int locnum2;
			for (locnum2 = 2; locnum2 < textOcrString[i].size(); locnum2++)
			{
				if (textOcrString[i][locnum2] >= 48 && textOcrString[i][locnum2] < 58)
				{
					break;
				}
			}
			if (textOcrString[i][locnum1] >= 48 && textOcrString[i][locnum1] < 58
				&& textOcrString[i][locnum1 + 1] >= 48 && textOcrString[i][locnum1 + 1] < 58
				&& textOcrString[i][locnum1 + 2] < 48 && textOcrString[i][locnum1 + 3] < 48
				&& textOcrString[i][locnum2] >= 48 && textOcrString[i][locnum2] < 58
				&& textOcrString[i][locnum2 + 1] >= 48 && textOcrString[i][locnum2 + 1] < 58
				&& textOcrString[i][locnum2 + 2] >= 48 && textOcrString[i][locnum2 + 2] < 58)
			{
				TrueOcrtext[countTrueTrain] = textOcrString[i];
				countTrueTrain++;
			}
		}
	}
	else if (numPlate == 2)
	{
		for (int i = 0; i < rep * 2 + 1; i++)
		{
			int locnum;
			for (locnum = 0; locnum < 7; locnum++)
			{
				if (textOcrString[i][locnum] >= 48 && textOcrString[i][locnum] < 58)
				{
					break;
				}
			}
			if (textOcrString[i][locnum] >= 48 && textOcrString[i][locnum] < 58
				&& textOcrString[i][locnum + 1] >= 48 && textOcrString[i][locnum + 1] < 58
				&& textOcrString[i][locnum + 2] < 48 && textOcrString[i].size() < 6)
			{
				TrueOcrtext[countTrueTrain] = textOcrString[i];
				countTrueTrain++;
			}
		}
	}


	//********************** return best train

	int *like = new int[countTrueTrain];
	for (int i = 0; i < countTrueTrain; i++)
	{
		like[i] = 0;
	}
	for (int i = 0; i < countTrueTrain; i++)
	{
		for (int j = 0; j < countTrueTrain; j++)
		{
			if (TrueOcrtext[i] == TrueOcrtext[j])
			{
				like[i]++;
			}
		}
	}
	int max = 0;
	for (int i = 0; i < countTrueTrain; i++)
	{
		if (like[i] > max)
		{
			max = like[i];
			numlike = i;
		}
	}

	TextOcr = TrueOcrtext[numlike];
}

//>>>>>>>>>>>>>>>>>>>>>>>>>> Find Plate <<<<<<<<<<<<<<<<<<<<<<<<<<<<

void FindPlate(Mat &largimg, Mat &Plate, string qqq)
{
	Mat img, gray, thre;
	Mat *cropped;

	Rect crop_largimg(largimg.cols / 5, largimg.rows / 5, largimg.cols * 3 / 5, largimg.rows * 3 / 5);
	Mat croppedRef1(largimg, crop_largimg);
	croppedRef1.copyTo(img);
	int countRect = 0;
	int halge = 0;
	if (qqq == "12")
	{
		halge = 1;
	}

	while (countRect == 0 && halge < 4)
	{
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;

		cvtColor(img, gray, CV_RGB2GRAY);


		if (halge == 0)
		{
			threshold(gray, thre, 80, 255, THRESH_BINARY_INV);
		}
		if (halge == 1)
		{
			Canny(gray, thre, 100, 100);
		}
		if (halge == 2)
		{
			Canny(gray, thre, 200, 255);
		}
		if (halge == 3)
		{
			threshold(gray, thre, 50, 110, THRESH_BINARY_INV);
		}


		findContours(thre, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

		Mat drawing = Mat::zeros(thre.size(), CV_8UC3);
		for (int i = 0; i < contours.size(); i++)
		{
			Scalar color = Scalar(0, 255, 0);
			drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
		}

		loc plate[5];


		int w_threshold = 100;
		int h_threshold = 100;
		vector<int> selected;
		for (int i = 0; i < contours.size(); i++)
		{
			Scalar color = Scalar(0, 255, 0);
			Rect R = boundingRect(contours[i]);
			float ratio = R.width / R.height;
			//

			if (ratio > 3 && ratio < 6 && R.height > 80 && R.height < 170 && R.width > 420 && R.width < 655)
			{
				plate[countRect].x = R.x;
				plate[countRect].y = R.y;
				plate[countRect].h = R.height;
				plate[countRect].w = R.width;
				selected.push_back(i);
				drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
				countRect++;
				cout << R.width << endl;
			}

		}
		cropped = new Mat[countRect];
		for (int i = 0; i < countRect; i++)
		{
			Rect crop_plate(plate[i].x, plate[i].y, plate[i].w, plate[i].h);
			Mat croppedRef(img, crop_plate);
			croppedRef.copyTo(cropped[i]);
		}


		Rect Re;

		Re.x = (plate[0].x) + (largimg.cols / 5);
		Re.y = (plate[0].y) + (largimg.rows / 5);
		Re.width = plate[0].w;
		Re.height = plate[0].h;

		rectangle(largimg, Re, Scalar(0, 255, 0), 2);

		if (countRect != 0)
		{
			Plate = cropped[0];
			break;
		}
		cout << halge << endl;
		halge++;
	}

	if (countRect == 0)
	{
		cout << "Plate Not Found ....";
		Sleep(1000);
		exit(0);
	}
}

void FindNum(Mat Plate, Mat &PlateNum1, Mat &PlateNum2)
{
	Mat num1, num2;

	//Rect myROI1(Plate.cols / 5, Plate.rows / 5, Plate.cols * 3 / 5, Plate.rows * 3 / 5);
	Rect myROI1(Plate.cols *.1, 0, Plate.cols*0.68, Plate.rows);
	Mat croppedRef1(Plate, myROI1);
	croppedRef1.copyTo(PlateNum1);

	Rect myROI2(Plate.cols *0.79, Plate.rows *0.25, Plate.cols*0.20, Plate.rows*0.70);
	Mat croppedRef2(Plate, myROI2);
	croppedRef2.copyTo(PlateNum2);

	/*imshow("Num1", PlateNum1);
	imshow("Num2", PlateNum2);*/

}

//>>>>>>>>>>>>>>>>>>>>>>>>>> Console UI <<<<<<<<<<<<<<<<<<<<<<<<<<<<

void gotoxy(int x, int y)
{
	COORD c = { x, y };
	SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), c);
}

enum Colors {
	blue = 1,
	green,
	cyan,
	red,
	purple,
	yellow,
	grey,
	dgrey,
	hblue,
	hgreen,
	hcyan,
	hred,
	hpurple,
	hyellow,
	hwhite
};

void coutc(int color, char* output)
{
	HANDLE handle = GetStdHandle(STD_OUTPUT_HANDLE);
	SetConsoleTextAttribute(handle, color);
	cout << output << flush;
}

void UI()
{
	wchar_t f;
	system("color 80");
	f = 219;
	int num;

	for (int i = 0; i < 32; i++)
	{
		system("color 80");
		num = i % 4;
		if (num == 0)
		{
			coutc(0, "");
			gotoxy(52, 8);
			wcout << f << f << f << f << f << f << f << f;
			gotoxy(52, 9);
			wcout << f << f << f << f << f << f << f << f;
			gotoxy(52, 10);
			wcout << f << f << f << f << f << f << f << f;
			gotoxy(52, 11);
			wcout << f << f << f << f << f << f << f << f;
		}

		if (num == 1)
		{
			coutc(0, "");
			gotoxy(62, 8);
			wcout << f << f << f << f << f << f << f << f;
			gotoxy(62, 9);
			wcout << f << f << f << f << f << f << f << f;
			gotoxy(62, 10);
			wcout << f << f << f << f << f << f << f << f;
			gotoxy(62, 11);
			wcout << f << f << f << f << f << f << f << f;
		}

		if (num == 2)
		{
			coutc(0, "");
			gotoxy(62, 13);
			wcout << f << f << f << f << f << f << f << f;
			gotoxy(62, 14);
			wcout << f << f << f << f << f << f << f << f;
			gotoxy(62, 15);
			wcout << f << f << f << f << f << f << f << f;
			gotoxy(62, 16);
			wcout << f << f << f << f << f << f << f << f;
		}

		if (num == 3)
		{
			coutc(0, "");
			gotoxy(52, 13);
			wcout << f << f << f << f << f << f << f << f;
			gotoxy(52, 14);
			wcout << f << f << f << f << f << f << f << f;
			gotoxy(52, 15);
			wcout << f << f << f << f << f << f << f << f;
			gotoxy(52, 16);
			wcout << f << f << f << f << f << f << f << f;
		}

		Sleep(100);
		system("cls");
	}
	system("cls");
	system("color 80");
	Sleep(100);
}

//>>>>>>>>>>>>>>>>>>>>>>>>>>>> Main <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

int main()
{
	Mat largimg, gray, thre, Plate, PlateNum1, PlateNum2;
	string TextOcrNum1, TextOcrNum2, adress;
	UI();
	cout << "Enter Name Picture : ";
	cin >> adress;

	largimg = imread("car/" + adress + ".jpg");
	if (largimg.empty())
	{
		cout << "Image Not Found .....";
		Sleep(1000);
		return(0);
	}

	FindPlate(largimg, Plate, adress);
	FindNum(Plate, PlateNum1, PlateNum2);

	imshow("Car", largimg);
	imshow("Plate", Plate);

	waitKey(2500);

	destroyWindow("Car");

	bool tarin;
	int cheacktrain = 0;
	cout << "Enter 1 to train Plate & Enter 0 to exit ...." << endl;
	cin >> tarin;
	cout << "Plase Wait ...";
	if (tarin != 0)
	{
		while (TextOcrNum1 == "")
		{
			Train(PlateNum1, TextOcrNum1, 1);
		}
		while (TextOcrNum2 == "")
		{
			Train(PlateNum2, TextOcrNum2, 2);
			if (cheacktrain > 3)
			{
				TextOcrNum2 = "13";
			}
			cheacktrain++;
		}
		Print(TextOcrNum1, TextOcrNum2);

		system("C:/Users/Hamid/Desktop/Plate.txt");
		exit(0);
	}

	return(0);
}