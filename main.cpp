#include <GL/freeglut.h>
#include <math.h>
#include <iostream>
#include <ctime>
#include <fstream>
#include <string>
#include "./Camera.h"
#include <omp.h>

#define METHOD_NUM 2
#define TRIANGLE_NUM 9
#define LIGHT_SOURCE_NUM 7
enum MouseMode {
    MM_CAMERA
};

struct RayHit {
    double t;
    double alpha;
    double beta;
    int mesh_idx;
};

struct TriangleObj {
    Eigen::Vector3d v1;
    Eigen::Vector3d v2;
    Eigen::Vector3d v3;
    Eigen::Vector3d n;
    Eigen::Vector3d color;
    std::string color_name;
    double area;
    bool is_light;
    double kd;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

const int g_FilmWidth = 640;
const int g_FilmHeight = 480;
const int sample_num = 50;

int mx, my;
int width = 640;
int height = 480;
int NUM_OF_SAMPLE = 0;
int *g_CountBuffer = nullptr;
int count_rayhit_idx[TRIANGLE_NUM];
Eigen::Vector3d sum;

const double __FAR__ = 1.0e33;

double intensity = 5;
double g_FrameSize_WindowSize_Scale_x = 1.0;
double g_FrameSize_WindowSize_Scale_y = 1.0;
double tri_area = 0.0;
double pdf[TRIANGLE_NUM];
double cdf[LIGHT_SOURCE_NUM];


float *g_FilmBuffer = nullptr;
float *g_AccumulationBuffer = nullptr;

bool g_DrawFilm = true;

GLuint g_FilmTexture = 0;
MouseMode g_MouseMode = MM_CAMERA;
Camera g_Camera;
TriangleObj tris[TRIANGLE_NUM];


inline float drand48() {
    return float(((double) (rand()) / (RAND_MAX))); /* RAND_MAX = 32767 */
}

//RGB値(0~255)を(0~1)へ正規化
Eigen::Vector3d rgbNormalize(const Eigen::Vector3d rgb) {
    Eigen::Vector3d out_rgb{rgb.x() / 255, rgb.y() / 255, rgb.z() / 255};
    return out_rgb;
}

void initFilm() {
    g_FilmBuffer = (float *) malloc(sizeof(float) * g_FilmWidth * g_FilmHeight * 3);
    memset(g_FilmBuffer, 0, sizeof(float) * g_FilmWidth * g_FilmHeight * 3);

    g_AccumulationBuffer = (float *) malloc(sizeof(float) * g_FilmWidth * g_FilmHeight * 3);
    g_CountBuffer = (int *) malloc(sizeof(int) * g_FilmWidth * g_FilmHeight);

    glGenTextures(1, &g_FilmTexture);
    glBindTexture(GL_TEXTURE_2D, g_FilmTexture);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, g_FilmWidth, g_FilmHeight, 0, GL_RGB, GL_FLOAT, g_FilmBuffer);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
}

void resetFilm() {
    memset(g_AccumulationBuffer, 0, sizeof(float) * g_FilmWidth * g_FilmHeight * 3);
    memset(g_CountBuffer, 0, sizeof(int) * g_FilmWidth * g_FilmHeight);
}


void updateFilm() {

    for (int i = 0; i < g_FilmWidth * g_FilmHeight; i++) {
        if (g_CountBuffer[i] > 0) {
            g_FilmBuffer[i * 3] = g_AccumulationBuffer[i * 3] / g_CountBuffer[i];
            g_FilmBuffer[i * 3 + 1] = g_AccumulationBuffer[i * 3 + 1] / g_CountBuffer[i];
            g_FilmBuffer[i * 3 + 2] = g_AccumulationBuffer[i * 3 + 2] / g_CountBuffer[i];
        } else {
            g_FilmBuffer[i * 3] = 0.0;
            g_FilmBuffer[i * 3 + 1] = 0.0;
            g_FilmBuffer[i * 3 + 2] = 0.0;
        }
    }
    glBindTexture(GL_TEXTURE_2D, g_FilmTexture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, g_FilmWidth, g_FilmHeight, GL_RGB, GL_FLOAT, g_FilmBuffer);
}

void drawFilm() {
    Eigen::Vector3d screen_center = g_Camera.getEyePoint() - g_Camera.getZVector() * g_Camera.getFocalLength();
    Eigen::Vector3d p1 = screen_center - g_Camera.getXVector() * g_Camera.getScreenWidth() * 0.5 - g_Camera.getYVector() * g_Camera.getScreenHeight() * 0.5;
    Eigen::Vector3d p2 = screen_center + g_Camera.getXVector() * g_Camera.getScreenWidth() * 0.5 - g_Camera.getYVector() * g_Camera.getScreenHeight() * 0.5;
    Eigen::Vector3d p3 = screen_center + g_Camera.getXVector() * g_Camera.getScreenWidth() * 0.5 + g_Camera.getYVector() * g_Camera.getScreenHeight() * 0.5;
    Eigen::Vector3d p4 = screen_center - g_Camera.getXVector() * g_Camera.getScreenWidth() * 0.5 + g_Camera.getYVector() * g_Camera.getScreenHeight() * 0.5;

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, g_FilmTexture);

    glBegin(GL_TRIANGLES);
    glColor3f(1.0, 1.0, 1.0);

    glTexCoord2f(0.0, 1.0);
    glVertex3f(p1.x(), p1.y(), p1.z());
    glTexCoord2f(1.0, 1.0);
    glVertex3f(p2.x(), p2.y(), p2.z());
    glTexCoord2f(1.0, 0.0);
    glVertex3f(p3.x(), p3.y(), p3.z());

    glTexCoord2f(0.0, 1.0);
    glVertex3f(p1.x(), p1.y(), p1.z());
    glTexCoord2f(1.0, 0.0);
    glVertex3f(p3.x(), p3.y(), p3.z());
    glTexCoord2f(0.0, 0.0);
    glVertex3f(p4.x(), p4.y(), p4.z());

    glEnd();

    glDisable(GL_TEXTURE_2D);
}

void clearRayTracedResult() {
    memset(g_FilmBuffer, 0, sizeof(float) * g_FilmWidth * g_FilmHeight * 3);
}

void makeTriangle(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2, const Eigen::Vector3d &v3, const Eigen::Vector3d &color, const std::string &color_name, const bool &is_light, const double &kd, TriangleObj &out_Tri) {
    out_Tri.v1 = v1;
    out_Tri.v2 = v2;
    out_Tri.v3 = v3;
    out_Tri.color = rgbNormalize(color);
    out_Tri.color_name = color_name;
    out_Tri.area = (((v2 - v1).cross(v3 - v1)) / 2).norm();
    out_Tri.is_light = is_light;
    out_Tri.kd = kd;
    out_Tri.n = ((v2 - v1).cross(v3 - v1)).normalized();
}

void rayTriangleIntersection(const TriangleObj &tris, const Ray &in_Ray, RayHit &out_Result) {
    out_Result.t = __FAR__;

    Eigen::Vector3d v1 = tris.v1;
    Eigen::Vector3d v2 = tris.v2;
    Eigen::Vector3d v3 = tris.v3;

    Eigen::Vector3d triangle_normal = ((v2 - v1).cross(v3 - v1)).normalized();


    double denominator = triangle_normal.dot(in_Ray.d);

    if (denominator >= 0.0)
        return;

    const double t = triangle_normal.dot(v3 - in_Ray.o) / denominator;
    if (t <= 0.0)
        return;

    const Eigen::Vector3d x = in_Ray.o + t * in_Ray.d;

    Eigen::Matrix<double, 3, 2> A;
    A.col(0) = v1 - v3;
    A.col(1) = v2 - v3;

    Eigen::Matrix2d ATA = A.transpose() * A;
    const Eigen::Vector2d b = A.transpose() * (x - v3);

    const Eigen::Vector2d alpha_beta = ATA.inverse() * b;

    //三角形の内部判定
    if (alpha_beta.x() < 0.0 || 1.0 < alpha_beta.x() || alpha_beta.y() < 0.0 ||
        1.0 < alpha_beta.y() || (1 - alpha_beta.x() - alpha_beta.y()) < 0.0 ||
        1.0 < (1 - alpha_beta.x() - alpha_beta.y()))
        return;

    out_Result.t = t;
    out_Result.alpha = alpha_beta.x();
    out_Result.beta = alpha_beta.y();
}

void TotalTriangleArea(double &t_area) {
    for (int i = 0; i < TRIANGLE_NUM; i++) {
        if (tris[i].is_light) {
            t_area += tris[i].area;
        }
    }
}

void PDF(double *PDF, double &t_area) {
    for (int i = 0; i < TRIANGLE_NUM; i++) {
        PDF[i] = tris[i].area / t_area;
    }
}

void CDF(double *CDF, double *PDF) {
    for (int i = 0; i < LIGHT_SOURCE_NUM; i++) {
        if (i == 0) {
            CDF[i] = PDF[i];
        } else {
            CDF[i] = CDF[i - 1] + PDF[i];
        }
    }
}

void CalcAdoptionRatio() {
    TotalTriangleArea(tri_area);
    PDF(pdf, tri_area);
    CDF(cdf, pdf);
}

void ChooseTriangle(int &t) {
    double random = drand48();
    auto Iter = std::lower_bound(std::begin(cdf), std::end(cdf), random);
    t = Iter - std::begin(cdf);
}


void rayTracing(const Ray &in_Ray, RayHit &io_Hit) {
    double t_min = __FAR__;
    double alpha_I = 0.0, beta_I = 0.0;
    int mesh_idx = -1;

    for (int i = 0; i < TRIANGLE_NUM; i++) {
        RayHit temp_hit{};
        rayTriangleIntersection(tris[i], in_Ray, temp_hit);
        if (temp_hit.t < t_min) {
            t_min = temp_hit.t;
            alpha_I = temp_hit.alpha;
            beta_I = temp_hit.beta;
            mesh_idx = i;
        }
    }

    io_Hit.t = t_min;
    io_Hit.alpha = alpha_I;
    io_Hit.beta = beta_I;
    io_Hit.mesh_idx = mesh_idx;
}

void directional_sampling(const Eigen::Vector3d &x, const RayHit &ray_hit, Eigen::Vector3d &pixel_color) {

    for (int n = 0; n < sample_num; n++) {
        const double theta = asin(sqrt(drand48()));
        const double phi = 2 * M_PI * drand48();
        const Eigen::Vector3d u = tris[ray_hit.mesh_idx].n;
        const Eigen::Vector3d v = u.y() < 1 ? u.cross(Eigen::Vector3d{0.0, 1.0, 0.0}).normalized() : u.cross(Eigen::Vector3d{1.0, 0.0, 0.0}).normalized();
        const Eigen::Vector3d w = v.cross(u);
        const Eigen::Vector3d omega_yup = {sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi)};
        const Eigen::Vector3d omega = omega_yup.y() * u + omega_yup.x() * v + omega_yup.z() * w;

        Ray _ray;
        _ray.o = x;
        _ray.d = omega;

        RayHit _ray_hit;

        rayTracing(_ray, _ray_hit);

        if (_ray_hit.mesh_idx != -1 && tris[_ray_hit.mesh_idx].is_light == true) {
            sum = sum + intensity * tris[_ray_hit.mesh_idx].color;
        }

    }
    pixel_color = tris[ray_hit.mesh_idx].kd * tris[ray_hit.mesh_idx].color.cwiseProduct(sum);
}

void surface_sampling(const Eigen::Vector3d &x, const double &totalTriangleArea, const RayHit &ray_hit, Eigen::Vector3d &pixel_color) {

    for (int i = 0; i < sample_num; i++) {
        int tri_idx;
        int V = 0;
        double cosx;
        double cosy = -1;
        Eigen::Vector3d xa;
        Eigen::Vector3d xa_x;
        Eigen::Vector3d w;
        Eigen::Vector3d ny;
        Eigen::Vector3d nx;

        const float gamma = 1.0 - sqrt(1.0 - drand48());
        const float beta = drand48() * (1.0 - gamma);

        while (cosy < 0) {
            ChooseTriangle(tri_idx);
            xa = (1.0 - beta - gamma) * tris[tri_idx].v1 + beta * tris[tri_idx].v2 + gamma * tris[tri_idx].v3;
            xa_x = xa - x;
            w = xa_x.normalized();
            ny = tris[tri_idx].n;
            nx = tris[ray_hit.mesh_idx].n;
            cosx = w.dot(nx);
            cosy = (-w).dot(ny);
        }

        Ray _ray_x_xa;
        _ray_x_xa.o = xa;
        _ray_x_xa.d = -w;

        RayHit _ray_hit_x_xa;

        rayTracing(_ray_x_xa, _ray_hit_x_xa);

        Ray _ray_xa_x;
        _ray_xa_x.o = x;
        _ray_xa_x.d = w;

        RayHit _ray_hit_xa_x;

        rayTracing(_ray_xa_x, _ray_hit_xa_x);

        if (_ray_hit_x_xa.mesh_idx == ray_hit.mesh_idx && _ray_hit_xa_x.mesh_idx == tri_idx) {
            V = 1;
        }

        sum += (intensity * tris[ray_hit.mesh_idx].kd / M_PI * cosx * cosy / xa_x.squaredNorm() * V) * tris[tri_idx].color;
    }

    sum *= totalTriangleArea;
    pixel_color = tris[ray_hit.mesh_idx].color.cwiseProduct(sum);
}

void saveppm() {

    std::ofstream writing_file;
    std::string filename;
    std::string path = "C:/Users/narus/";

    if (METHOD_NUM == 1)
        filename = path + "method1_" + std::to_string(NUM_OF_SAMPLE) + ".ppm";
    else
        filename = path + "method2_" + std::to_string(NUM_OF_SAMPLE) + ".ppm";

    writing_file.open(filename, std::ios::out);
    std::string header = "P3\n" + std::to_string(g_FilmWidth) + " " + std::to_string(g_FilmHeight) + "\n" + "255\n";
    writing_file << header << std::endl;

    for (int i = 0; i < g_FilmWidth * g_FilmHeight; i++) {
        std::string pixel = std::to_string(int(g_FilmBuffer[i * 3] * 255)) + " " + std::to_string(int(g_FilmBuffer[i * 3 + 1] * 255)) + " " + std::to_string(int(g_FilmBuffer[i * 3 + 2] * 255));
        if (i % g_FilmWidth == g_FilmWidth - 1) {
            writing_file << pixel << std::endl;
        } else {
            writing_file << pixel << " ";
        }
    }
    writing_file.close();
    std::cout << "Number of sample =" << NUM_OF_SAMPLE << "ppm file saved !\n";
}


void Learn() {

    memset(count_rayhit_idx, 0, sizeof count_rayhit_idx);
    double totalTriangleArea = 0.0;
    TotalTriangleArea(totalTriangleArea);
    for (int Y = 0; Y < g_FilmHeight; Y++) {
        for (int X = 0; X < g_FilmWidth; X++) {
            const int pixel_idx = Y * g_FilmWidth + X;
            const double p_x = (X + 0.5) / g_FilmWidth;
            const double p_y = (Y + 0.5) / g_FilmHeight;

            Ray ray;
            g_Camera.screenView(p_x, p_y, ray);

            RayHit ray_hit;
            rayTracing(ray, ray_hit);

            if (ray_hit.mesh_idx >= 0) {
                if (tris[ray_hit.mesh_idx].is_light == true) {
                    for (int i = 0; i < 3; i++) {
                        g_AccumulationBuffer[pixel_idx * 3 + i] += tris[ray_hit.mesh_idx].color[i];
                    }
                    g_CountBuffer[pixel_idx] += 1;

                } else {
                    //飛ばされたレイと拡散面上の点
                    const Eigen::Vector3d x = ray.o + ray_hit.t * ray.d;
                    Eigen::Vector3d pixel_color;
                    Eigen::Vector3d I_n;
                    sum = {0.0, 0.0, 0.0};

                    //方向サンプリング
                    if (METHOD_NUM == 1)
                        directional_sampling(x, ray_hit, pixel_color);

                        //面サンプリング
                    else
                        surface_sampling(x, totalTriangleArea, ray_hit, pixel_color);

                    //求めたRGB値をピクセルごとに代入
                    for (int i = 0; i < 3; i++) {
                        g_AccumulationBuffer[pixel_idx * 3 + i] += pixel_color[i];
                    }
                    g_CountBuffer[pixel_idx] += sample_num;
                    NUM_OF_SAMPLE = g_CountBuffer[pixel_idx];

                }
            } else {
                const Eigen::Vector3d pixel_color = rgbNormalize(Eigen::Vector3d{40, 40, 40});
                for (int i = 0; i < 3; i++) {
                    g_AccumulationBuffer[pixel_idx * 3 + i] += pixel_color[i];
                }
                g_CountBuffer[pixel_idx] += 1;
            }
        }
    }
    updateFilm();
    glutPostRedisplay();

}

void drawTriangleObj(const TriangleObj &rect) {
    glBegin(GL_TRIANGLES);
    glColor3f(rect.color(0), rect.color(1), rect.color(2));

    glVertex3f(rect.v1.x(), rect.v1.y(), rect.v1.z());
    glVertex3f(rect.v2.x(), rect.v2.y(), rect.v2.z());
    glVertex3f(rect.v3.x(), rect.v3.y(), rect.v3.z());

    glEnd();
}

void mouseDrag(int x, int y) {
    int _dx = x - mx, _dy = y - my;
    mx = x;
    my = y;

    double dx = double(_dx) / double(width);
    double dy = -double(_dy) / double(height);

    if (g_MouseMode == MM_CAMERA) {
        double scale = 2.0;

        g_Camera.rotateCameraInLocalFrameFixLookAt(dx * scale);
        resetFilm();
        updateFilm();
        glutPostRedisplay();
    }
}

void mouseDown(int x, int y) {
    mx = x;
    my = y;
}

void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
        mouseDown(x, y);
}

void key(unsigned char key, int x, int y) {
    switch (key) {
        case 'C':
        case 'c':
            g_MouseMode = MM_CAMERA;
            break;
        case 'f':
        case 'F':
            g_DrawFilm = !g_DrawFilm;
            glutPostRedisplay();
            break;
        case 'p':
        case 'P':
            saveppm();
            break;
    }
}

void projection_and_modelview(const Camera &in_Camera) {
    const double fovy_deg = (2.0 * 180.0 / M_PI) *
                            atan(0.024 * 0.5 / in_Camera.getFocalLength());

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(fovy_deg, double(width) / double(height), 0.01 * in_Camera.getFocalLength(), 1000.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    const Eigen::Vector3d lookAtPoint = in_Camera.getLookAtPoint();
    gluLookAt(in_Camera.getEyePoint().x(), in_Camera.getEyePoint().y(), in_Camera.getEyePoint().z(), lookAtPoint.x(), lookAtPoint.y(), lookAtPoint.z(), in_Camera.getYVector().x(), in_Camera.getYVector().y(), in_Camera.getYVector().z());
}

void display() {
    glViewport(0, 0, width * g_FrameSize_WindowSize_Scale_x, height * g_FrameSize_WindowSize_Scale_y);

    glClearColor(0.0, 0.0, 1.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    projection_and_modelview(g_Camera);

    glEnable(GL_DEPTH_TEST);

    Learn();

    if (g_DrawFilm)
        drawFilm();

    for (int i = 0; i < TRIANGLE_NUM; i++) {
        drawTriangleObj(tris[i]);
    }

    glDisable(GL_DEPTH_TEST);

    glutSwapBuffers();
}

void resize(int w, int h) {
    width = w;
    height = h;
}

int main(int argc, char *argv[]) {
    g_Camera.setEyePoint(Eigen::Vector3d{6.0, 1.5, 1.0});
    g_Camera.lookAt(Eigen::Vector3d{0.0, 0.5, -1.25}, Eigen::Vector3d{0.0, 1.0, 0.0});

    glutInit(&argc, argv);
    glutInitWindowSize(width, height);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_MULTISAMPLE);

    glutCreateWindow("Hello world!!");

    // With retina display, frame buffer size is twice the window size.
    // Viewport size should be set on the basis of the frame buffer size, rather than the window size.
    // g_FrameSize_WindowSize_Scale_x and g_FrameSize_WindowSize_Scale_y account for this factor.
    GLint dims[4] = {0};
    glGetIntegerv(GL_VIEWPORT, dims);
    g_FrameSize_WindowSize_Scale_x = double(dims[2]) / double(width);
    g_FrameSize_WindowSize_Scale_y = double(dims[3]) / double(height);

    makeTriangle(Eigen::Vector3d{-2.0, 2.25, -2.0}, Eigen::Vector3d{-3.0, 0.0, -2.0}, Eigen::Vector3d{0.0, 0.0, -2.0}, Eigen::Vector3d{215.0, 14.0, 74.0}, "Red", true, 1.0, tris[0]);
    makeTriangle(Eigen::Vector3d{0.25, 0.0, -2.0}, Eigen::Vector3d{0.5, 1.5, -2.0}, Eigen::Vector3d{-1.5, 2.25, -2.0}, Eigen::Vector3d{1.0, 195.0, 215.0}, "Blue", true, 1.0, tris[1]);
    makeTriangle(Eigen::Vector3d{1.5, 2.5, -2.0}, Eigen::Vector3d{0.75, 1.0, -2.0}, Eigen::Vector3d{1.75, 0.0, -2.0}, Eigen::Vector3d{50.0, 205.0, 50.0}, "Green", true, 1.0, tris[2]);
    makeTriangle(Eigen::Vector3d{0.75, 0.75, -2.0}, Eigen::Vector3d{0.5, 0.0, -2.0}, Eigen::Vector3d{1.5, 0.0, -2.0}, Eigen::Vector3d{255.0, 244.0, 1.0}, "Yellow", true, 1.0, tris[3]);
    makeTriangle(Eigen::Vector3d{2.0, 1.5, -2.0}, Eigen::Vector3d{2.0, 0.0, -2.0}, Eigen::Vector3d{3.0, 0.0, -2.0}, Eigen::Vector3d{147.0, 112.0, 219.0}, "Purple", true, 1.0, tris[4]);
    makeTriangle(Eigen::Vector3d{0.5, 0.25, 0.5}, Eigen::Vector3d{0.25, 0.0, 0.5}, Eigen::Vector3d{0.75, 0.0, 0.25}, Eigen::Vector3d{255.0, 255.0, 255.0}, "White", true, 1.0, tris[5]);
    makeTriangle(Eigen::Vector3d{-1.0, 1.5, 1.25}, Eigen::Vector3d{-1.0, 0.0, 1.25}, Eigen::Vector3d{-1.75, 0.0, 0.5}, Eigen::Vector3d{210.0, 105.0, 30.0}, "Orange", true, 1.0, tris[6]);
    makeTriangle(Eigen::Vector3d{0.0, 0.0, 1.5}, Eigen::Vector3d{3.0, 0.0, -1.5}, Eigen::Vector3d{-3.0, 0.0, -1.5}, Eigen::Vector3d{255.0, 255.0, 255.0}, "Diffusion_Surface_1", false, 1.0, tris[7]);
    makeTriangle(Eigen::Vector3d{-1.0, 0.0, -1.0}, Eigen::Vector3d{-0.5, 0.5, -1.0}, Eigen::Vector3d{-1.5, 0.5, -1.0}, Eigen::Vector3d{255.0, 255.0, 255.0}, "Diffusion_Surface_2", false, 1.0, tris[8]);

    CalcAdoptionRatio();

    std::srand(time(NULL));

    glutDisplayFunc(display);
    glutReshapeFunc(resize);
    glutMouseFunc(mouse);
    glutMotionFunc(mouseDrag);
    glutKeyboardFunc(key);
    initFilm();
    clearRayTracedResult();
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    glutMainLoop();
    return 0;
}