#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define RADIUS 50

using namespace cv;
using namespace std;

int gl_slider = 0;
int gl_slider_max = 100;

int gh_slider = 50;
int gh_slider_max = 100;

int c_slider = 5;
int c_slider_max = 100;

int d0_slider = 5;
int d0_slider_max = 200;

Mat input_img;
Mat padded;
Mat_<float> realInput, zeros;
Mat complexImage;
Mat filter, tmp;
vector<Mat> planos;

int dft_M, dft_N;

void deslocaDFT(Mat& image);
Mat create_homomorfic_filter(Size paddedSize, double gl, double gh, double c, double d0);
void on_trackbar_move(int, void*);

int main(int argc, char** argv) {

    input_img = imread("C:\\Users\\mateu\\Downloads\\piano.jpeg", CV_LOAD_IMAGE_GRAYSCALE);
    resize(input_img, input_img, Size(800, 600));
    imshow("Original", input_img);

    // identifica os tamanhos otimos para
    // calculo do FFT
    dft_M = getOptimalDFTSize(input_img.rows);
    dft_N = getOptimalDFTSize(input_img.cols);

    // realiza o padding da imagem
    copyMakeBorder(input_img, padded, 0,
        dft_M - input_img.rows, 0,
        dft_N - input_img.cols,
        BORDER_CONSTANT, Scalar::all(0));

    zeros = Mat_<float>::zeros(padded.size());

    char TrackbarName[50];

    
    namedWindow("a", WINDOW_NORMAL);

    
    sprintf_s(TrackbarName, "Gamma L x %d", gl_slider_max);
    createTrackbar(TrackbarName, "a", &gl_slider, gl_slider_max, on_trackbar_move);

    sprintf_s(TrackbarName, "Gamma H x %d", gh_slider_max);
    createTrackbar(TrackbarName, "a", &gh_slider, gh_slider_max, on_trackbar_move);

    sprintf_s(TrackbarName, "C x %d", c_slider_max);
    createTrackbar(TrackbarName, "a", &c_slider, c_slider_max, on_trackbar_move);

    sprintf_s(TrackbarName, "Cutoff Frequency x %d", d0_slider_max);
    createTrackbar(TrackbarName, "a", &d0_slider, d0_slider_max, on_trackbar_move);

    on_trackbar_move(0, NULL);
    


    waitKey(0);
    return 0;
}



//troca os quadrantes da imagem da DFT
void deslocaDFT(Mat& image) {
    Mat tmp, A, B, C, D;

    //se a imagem tiver tamanho impar, recorta a regiao para
    //evitar cópias de tamanho desigual
    image = image(Rect(0, 0, image.cols & -2, image.rows & -2));
    int cx = image.cols / 2;
    int cy = image.rows / 2;

    //reorganizacao dos quadrantes da transformada
    //A B   ->  D C
    //C D       B A
    A = image(Rect(0, 0, cx, cy));
    B = image(Rect(cx, 0, cx, cy));
    C = image(Rect(0, cy, cx, cy));
    D = image(Rect(cx, cy, cx, cy));

    // A <-> D
    A.copyTo(tmp);  D.copyTo(A);  tmp.copyTo(D);

    // C <-> B
    C.copyTo(tmp);  B.copyTo(C);  tmp.copyTo(B);
}


//cria filtro homomorfico
cv::Mat create_homomorfic_filter(cv::Size paddedSize, double gl, double gh, double c, double d0) {
    Mat filter = Mat(paddedSize, CV_32FC2, Scalar(0));
    Mat tmp = Mat(dft_M, dft_N, CV_32F);

    for (int i = 0; i < tmp.rows; i++) {
        for (int j = 0; j < tmp.cols; j++) {
            float coef = (i - dft_M / 2) * (i - dft_M / 2) + (j - dft_N / 2) * (j - dft_N / 2);
            tmp.at<float>(i, j) = (gh - gl) * (1.0 - (float)exp(-(c * coef / (d0 * d0)))) + gl;
        }
    }

    // cria a matriz com as componentes do filtro e junta
    // ambas em uma matriz multicanal complexa  
    Mat comps[] = { tmp, tmp };
    merge(comps, 2, filter);
    return filter;
}

void on_trackbar_move(int, void*) {
    // limpa o array de matrizes que vao compor a
    // imagem complexa
    planos.clear();

    // cria a compoente real e imaginaria (zeros)
    realInput = Mat_<float>(padded);
    //realInput += Scalar::all(1);
    //log(realInput,realInput);

    // insere as duas componentes no array de matrizes
    planos.push_back(realInput);
    planos.push_back(zeros);

    // combina o array de matrizes em uma unica
    // componente complexa
    // prepara a matriz complexa para ser preenchida
    complexImage = Mat(padded.size(), CV_32FC2, Scalar(0));
    merge(planos, complexImage);

    // calcula o dft
    dft(complexImage, complexImage);

    // realiza a troca de quadrantes
    deslocaDFT(complexImage);
    resize(complexImage, complexImage, padded.size());
    normalize(complexImage, complexImage, 0, 1, CV_MINMAX);

    // aplica o filtro frequencial
    float gl = (float)gl_slider / 100.0;
    float gh = (float)gh_slider / 100.0;
    float d0 = 25.0 * d0_slider / 100.0;
    float c = (float)c_slider / 100.0;

    cout << "gl = " << gl << endl;
    cout << "gh = " << gh << endl;
    cout << "d0 = " << d0 << endl;
    cout << "c = " << c << endl;

    Mat filter = create_homomorfic_filter(padded.size(), gl, gh, c, d0);
    mulSpectrums(complexImage, filter, complexImage, 0);

    // troca novamente os quadrantes
    deslocaDFT(complexImage);

    // calcula a DFT inversa
    idft(complexImage, complexImage);

    // limpa o array de planos
    planos.clear();

    // separa as partes real e imaginaria da
    // imagem filtrada
    split(complexImage, planos);

    // // normaliza a parte real para exibicao
    normalize(planos[0], planos[0], 0, 1, CV_MINMAX);
    resize(planos[0], planos[0], Size(800, 600));
    imshow("Filtro Homomórfico", planos[0]);
    namedWindow("Filtro Homomórfico", WINDOW_NORMAL);
}
