//
// gl_main.cpp
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include "GL/glew.h"
#include "GL/freeglut.h"
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>
#include <map>
// #include "GIPC.cuh"
#include "device_launch_parameters.h"
#include "mlbvh.cuh"
#include <stdio.h>
#include "load_mesh.h"
#include "cuda_tools/cuda_tools.h"
#include <queue>
//#include "timer.h"
#include "femEnergy.cuh"
#include "gpu_eigen_libs.cuh"
#include "fem_parameters.h"
#include "gipc_path.h"
#include <gipc/type_define.h>
#include <filesystem>
#include <gipc/statistics.h>
#include <gipc/utils/simple_scene_importer.h>
#include <Eigen/Geometry>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include <GIPC.cuh>
#include <gipc/utils/benchmark.h>
#include <iomanip>
#include <numeric>
#include <stdexcept>
#include <cmath>
#include <algorithm>

auto             assets_dir = std::string{gipc::assets_dir()};
std::string      metis_dir  = assets_dir + "sorted_mesh/";
double           collision_detection_buff_scale = 1;
double           motion_rate                    = 1;
double           linear_system_buff_scale       = 1.0;
mesh_obj         obj;
lbvh_f           bvh_f;
lbvh_e           bvh_e;
GIPC             ipc;
device_TetraData d_tetMesh;
tetrahedra_obj   tetMesh;
std::vector<Node>     nodes;
std::vector<AABB>     bvs;
std::vector<std::string>   obj_pathes;
int              initPath = 0;
int   step      = 0;
int   frameId   = 0;
int   surfNumId = 0;
float xRot      = 0.0f;
float yRot      = 0.f;
float xTrans    = 0;
float yTrans    = 0;
float zTrans    = 0;
int   ox;
int   oy;
int   buttonState;
float xRotLength    = 0.0f;
float yRotLength    = 0.0f;
float window_width  = 1000;
float window_height = 1000;
int   s_dimention   = 3;
bool  saveSurface   = false;
bool  change        = false;
bool  screenshot    = false;

bool isSetShader = false;

bool drawbvh     = false;
bool drawSurface = true;

bool stop = true;

double3 center;
double3 Ssize;

GLuint PN_vbo_;
GLuint VAO;
GLuint color_vbo_;
//GLuint color_vao_;
GLuint normal_vbo_;
//GLuint normal_vao_;
GLuint v;
GLuint f;
GLuint shaderProgram;

int            clothFaceOffset = 0;
int            bodyVertOffset  = 0;
double         global_offset   = 1.0;
std::vector<std::string> files;
std::vector<int>    file_vert_offsets;
std::vector<int>    file_tet_offsets;

void Init_CUDA()
{
    cudaError_t cudaStatus = cudaSetDevice(0);
    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        exit(0);
    }
}

#pragma pack(push, 1)
typedef struct
{
    uint16_t bfType;
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
} mBITMAPFILEHEADER;
#pragma pack(pop)

#pragma pack(push, 1)
typedef struct
{
    uint32_t biSize;
    int32_t  biWidth;
    int32_t  biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    int32_t  biXPelsPerMeter;
    int32_t  biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
} mBITMAPINFOHEADER;
#pragma pack(pop)

bool WriteBitmapFile(int width, int height, const std::string& file_name, unsigned char* bitmapData)
{
    mBITMAPFILEHEADER bitmapFileHeader;
    memset(&bitmapFileHeader, 0, sizeof(mBITMAPFILEHEADER));
    bitmapFileHeader.bfSize = sizeof(mBITMAPFILEHEADER);
    bitmapFileHeader.bfType = 0x4d42;  //BM
    bitmapFileHeader.bfOffBits = sizeof(mBITMAPFILEHEADER) + sizeof(mBITMAPINFOHEADER);

    mBITMAPINFOHEADER bitmapInfoHeader;
    memset(&bitmapInfoHeader, 0, sizeof(mBITMAPINFOHEADER));
    bitmapInfoHeader.biSize        = sizeof(mBITMAPINFOHEADER);
    bitmapInfoHeader.biWidth       = width;
    bitmapInfoHeader.biHeight      = height;
    bitmapInfoHeader.biPlanes      = 1;
    bitmapInfoHeader.biBitCount    = 24;
    bitmapInfoHeader.biCompression = 0L;
    bitmapInfoHeader.biSizeImage   = width * std::abs(height) * 3;

    //////////////////////////////////////////////////////////////////////////
    FILE*         filePtr;
    unsigned char tempRGB;
    int           imageIdx;

    for(imageIdx = 0; imageIdx < (int)bitmapInfoHeader.biSizeImage; imageIdx += 3)
    {
        tempRGB                  = bitmapData[imageIdx];
        bitmapData[imageIdx]     = bitmapData[imageIdx + 2];
        bitmapData[imageIdx + 2] = tempRGB;
    }

    filePtr = fopen(file_name.c_str(), "wb");
    if(NULL == filePtr)
    {
        return false;
    }

    fwrite(&bitmapFileHeader, sizeof(mBITMAPFILEHEADER), 1, filePtr);

    fwrite(&bitmapInfoHeader, sizeof(mBITMAPINFOHEADER), 1, filePtr);

    fwrite(bitmapData, bitmapInfoHeader.biSizeImage, 1, filePtr);

    fclose(filePtr);
    return true;
}

void SaveScreenShot(int width, int height, const std::string& file_name)
{
    int   data_len    = height * width * 3;  // bytes
    void* screen_data = malloc(data_len);
    memset(screen_data, 0, data_len);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, screen_data);
    WriteBitmapFile(width, height, file_name + ".bmp", (unsigned char*)screen_data);
    free(screen_data);
}

void saveSurfaceMesh(const std::string& path)
{
    std::stringstream ss;
    ss << path;
    ss.fill('0');
    ss.width(5);
    ss << (surfNumId++) / 1;  // / 10;
    //if (surfNumId % 10 != 0) return;
    ss << ".obj";
    std::string file_path = ss.str();
    std::ofstream    outSurf(file_path);

    std::map<int, int> meshToSurf;
    outSurf << "s 1" << std::endl;
    for(int i = 0; i < tetMesh.surfVerts.size(); i++)
    {
        const auto& pos = tetMesh.vertexes[tetMesh.surfVerts[i]];
        outSurf << "v " << pos.x << " " << pos.y << " " << pos.z << std::endl;
        meshToSurf[tetMesh.surfVerts[i]] = i;
    }

    for(int i = 0; i < tetMesh.surface.size(); i++)
    {
        const auto& tri = tetMesh.surface[i];
        outSurf << "f " << meshToSurf[tri.x] + 1 << " " << meshToSurf[tri.y] + 1
                << " " << meshToSurf[tri.z] + 1 << std::endl;
    }
    outSurf.close();
}


void saveTets(const std::string& path)
{
    int tetIdoffset = 0;
    for(int ii = 0; ii < 4096; ii++)
    {
        //tetMesh.output_tetrahedraMesh
        std::stringstream ss;
        ss << path;
        ss << ii;  // / 10;
        //if (surfNumId % 10 != 0) return;
        ss << ".msh";
        std::string file_path = ss.str();
        std::ofstream    outmsh1(file_path);

        std::map<int, int> meshToSurf;
        //outSurf << "s 1" << std::endl;
        outmsh1 << "$Nodes\n";
        outmsh1 << file_vert_offsets[ii + 1] - file_vert_offsets[ii] << std::endl;
        for(int i = 0; i < file_vert_offsets[ii + 1] - file_vert_offsets[ii]; i++)
        {
            const auto& pos = tetMesh.vertexes[i + file_vert_offsets[ii]];
            outmsh1 << i + 1 << " " << pos.x << " " << pos.y << " " << pos.z << std::endl;
            meshToSurf[i + file_vert_offsets[ii]] = i;
        }
        outmsh1 << "$Elements\n";
        outmsh1 << file_tet_offsets[ii + 1] << std::endl;

        for(int i = 0; i < file_tet_offsets[ii + 1]; i++)
        {
            int tetId = i + tetIdoffset;
            outmsh1 << i + 1 << " 4 0 " << meshToSurf[tetMesh.tetrahedras[tetId].x] + 1
                    << " " << meshToSurf[tetMesh.tetrahedras[tetId].y] + 1
                    << " " << meshToSurf[tetMesh.tetrahedras[tetId].z] + 1 << " "
                    << meshToSurf[tetMesh.tetrahedras[tetId].w] + 1 << std::endl;
        }
        tetIdoffset += file_tet_offsets[ii + 1];
        outmsh1.close();
    }
}

void draw_box2D(float ox, float oy, float width, float height)
{
    glLineWidth(2.5f);
    glColor3f(0.8f, 0.8f, 0.8f);

    glBegin(GL_LINES);

    glVertex3f(ox, oy, 0);
    glVertex3f(ox + width, oy, 0);

    glVertex3f(ox, oy, 0);
    glVertex3f(ox, oy + height, 0);

    glVertex3f(ox + width, oy, 0);
    glVertex3f(ox + width, oy + height, 0);

    glVertex3f(ox + width, oy + height, 0);
    glVertex3f(ox, oy + height, 0);

    glEnd();
}

void draw_box3D(float ox, float oy, float oz, float width, float height, float length, int boxType = 0)
{
    glLineWidth(0.5f);
    glColor3f(0.8f, 0.8f, 0.1f);
    if(boxType == 1)
    {
        glLineWidth(1.5f);
        glColor3f(0.8f, 0.8f, 0.8f);
    }
    glBegin(GL_LINES);

    glVertex3f(ox, oy, oz);
    glVertex3f(ox + width, oy, oz);

    glVertex3f(ox, oy, oz);
    glVertex3f(ox, oy + height, oz);

    glVertex3f(ox, oy, oz);
    glVertex3f(ox, oy, oz + length);

    glVertex3f(ox + width, oy, oz);
    glVertex3f(ox + width, oy + height, oz);

    glVertex3f(ox + width, oy + height, oz);
    glVertex3f(ox, oy + height, oz);

    glVertex3f(ox, oy + height, oz + length);
    glVertex3f(ox, oy, oz + length);

    glVertex3f(ox, oy + height, oz + length);
    glVertex3f(ox, oy + height, oz);

    glVertex3f(ox + width, oy, oz);
    glVertex3f(ox + width, oy, oz + length);

    glVertex3f(ox, oy, oz + length);
    glVertex3f(ox + width, oy, oz + length);

    glVertex3f(ox + width, oy + height, oz);
    glVertex3f(ox + width, oy + height, oz + length);

    glVertex3f(ox + width, oy + height, oz + length);
    glVertex3f(ox + width, oy, oz + length);

    glVertex3f(ox, oy + height, oz + length);
    glVertex3f(ox + width, oy + height, oz + length);

    glEnd();
}

void draw_lines(float ox, float oy, float oz, float width, float height, float length)
{
    glLineWidth(0.5f);
    glColor3f(0.8f, 0.8f, 0.8f);

    glBegin(GL_LINES);
    int numbers = 20;
    for(int i = 0; i <= numbers; i++)
    {
        //glVertex3f(ox, oy, oz);
        glVertex3f(ox + width * i / numbers, oy, 0);
        glVertex3f(ox + width * i / numbers, oy + height, 0);
    }

    for(int i = 0; i <= numbers; i++)
    {
        //glVertex3f(ox, oy, oz);
        glVertex3f(ox, oy + height * i / numbers, 0);
        glVertex3f(ox + width, oy + height * i / numbers, 0);
    }

    glEnd();


    glLineWidth(1.5f);
    glColor3f(0.8f, 0.8f, 0.f);
    glBegin(GL_LINES);
    glVertex3f(ox + width / 2, oy, 0);
    glVertex3f(ox + width / 2, oy + height, 0);

    glVertex3f(ox, oy + height / 2, 0);
    glVertex3f(ox + width, oy + height / 2, 0);

    glEnd();
}

void draw_mesh3D()
{
    glEnable(GL_DEPTH_TEST);
    glLineWidth(1.5f);
    glColor3f(0.9f, 0.1f, 0.1f);
    const std::vector<uint3>& surf = tetMesh.surface;  //obj.faces;
    glBegin(GL_TRIANGLES);


    for(int j = 0; j < tetMesh.surface.size(); j++)
    {
        glVertex3f((tetMesh.vertexes[surf[j].x].x),
                   (tetMesh.vertexes[surf[j].x].y),
                   (tetMesh.vertexes[surf[j].x].z));
        glVertex3f((tetMesh.vertexes[surf[j].y].x),
                   (tetMesh.vertexes[surf[j].y].y),
                   (tetMesh.vertexes[surf[j].y].z));
        glVertex3f((tetMesh.vertexes[surf[j].z].x),
                   (tetMesh.vertexes[surf[j].z].y),
                   (tetMesh.vertexes[surf[j].z].z));
    }
    glEnd();

    glColor3f(0.9f, 0.9f, 0.9f);
    //glDisable(GL_DEPTH_TEST);
    glLineWidth(0.1f);
    glBegin(GL_LINES);

    for(int j = 0; j < tetMesh.surfEdges.size(); j++)
    {
        //if ((tetMesh.surfEdges[j].x == 870 && tetMesh.surfEdges[j].y == 965) || (tetMesh.surfEdges[j].x == 965 && tetMesh.surfEdges[j].y == 870)) {
        //    glColor3f(0.9f, 0.1f, 0.1f);
        //    glLineWidth(3.4f);
        //}
        //else if ((tetMesh.surfEdges[j].x == 870 && tetMesh.surfEdges[j].y == 905) || (tetMesh.surfEdges[j].x == 905 && tetMesh.surfEdges[j].y == 870)) {
        //    glColor3f(0.9f, 0.9f, 0.1f);
        //    glLineWidth(3.4f);
        //}

        glVertex3f((tetMesh.vertexes[tetMesh.surfEdges[j].x].x),
                   (tetMesh.vertexes[tetMesh.surfEdges[j].x].y),
                   (tetMesh.vertexes[tetMesh.surfEdges[j].x].z));
        glVertex3f((tetMesh.vertexes[tetMesh.surfEdges[j].y].x),
                   (tetMesh.vertexes[tetMesh.surfEdges[j].y].y),
                   (tetMesh.vertexes[tetMesh.surfEdges[j].y].z));

        glColor3f(0.9f, 0.9f, 0.9f);
        glLineWidth(0.1f);
    }
    glEnd();

    //glColor3f(0.99f, 0.1f, 0.1f);
    ////glDisable(GL_DEPTH_TEST);
    //glPointSize(8);
    //glBegin(GL_POINTS);
    //glVertex3f((tetMesh.vertexes[2189].x), (tetMesh.vertexes[2189].y), (tetMesh.vertexes[2189].z));
    //glColor3f(0.99f, 0.99f, 0.1f);
    //glVertex3f((tetMesh.vertexes[870].x), (tetMesh.vertexes[870].y), (tetMesh.vertexes[870].z));
    //glVertex3f((tetMesh.vertexes[905].x), (tetMesh.vertexes[905].y), (tetMesh.vertexes[905].z));
    //glVertex3f((tetMesh.vertexes[965].x), (tetMesh.vertexes[965].y), (tetMesh.vertexes[965].z));
    //glEnd();
}

void draw_bvh()
{
    int num = (bvs.size() + 1) / 2;
    for(int j = 0; j < bvs.size(); j++)
    {
        int   i = j;
        float ox, oy, oz, bwidth, bheight, blength;
        ox      = (bvs[i].lower.x);
        oy      = (bvs[i].lower.y);
        oz      = (bvs[i].lower.z);
        bwidth  = (bvs[i].upper.x - bvs[i].lower.x);
        bheight = (bvs[i].upper.y - bvs[i].lower.y);
        blength = (bvs[i].upper.z - bvs[i].lower.z);
        draw_box3D(ox, oy, oz, bwidth, bheight, blength);
    }
}

int            counttt = 0;
std::vector<float3> getRenderGeometry(int& number)
{

    std::vector<double3> meshNormal(tetMesh.vertexNum, make_double3(0, 0, 0));
    number = tetMesh.surface.size();  //meshTemp.surfaceRender.size();
    std::vector<float3> pos_normal_color(3 * number * 3);

    for(int i = 0; i < number; i++)
    {
        //int tetId = meshTemp.surfaceRender[i][3];
        int v0 = tetMesh.surface[i].x;
        int v1 = tetMesh.surface[i].y;
        int v2 = tetMesh.surface[i].z;
        double3 vt0 = tetMesh.vertexes[v0];  // Vector3d(meshTemp.vertexes[v0][0], meshTemp.vertexes[v0][1], meshTemp.vertexes[v0][2]);
        double3 vt1 = tetMesh.vertexes[v1];  // Vector3d(meshTemp.vertexes[v1][0], meshTemp.vertexes[v1][1], meshTemp.vertexes[v1][2]);
        double3 vt2 = tetMesh.vertexes[v2];  // Vector3d(meshTemp.vertexes[v2][0], meshTemp.vertexes[v2][1], meshTemp.vertexes[v2][2]);
        double3 vec1 = __GEIGEN__::__minus(vt1, vt0);  //vt1 - vt0;
        double3 vec2 = __GEIGEN__::__minus(vt2, vt0);
        double3 normal =
            __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(vec1, vec2));  //vec1.cross(vec2).normalized();

        pos_normal_color[i * 9]     = make_float3(vt0.x, vt0.y, vt0.z);
        pos_normal_color[i * 9 + 3] = make_float3(vt1.x, vt1.y, vt1.z);
        pos_normal_color[i * 9 + 6] = make_float3(vt2.x, vt2.y, vt2.z);

        pos_normal_color[i * 9 + 2] = make_float3(0.6875f, 0.51953f, 0.38671f);
        pos_normal_color[i * 9 + 5] = make_float3(0.6875f, 0.51953f, 0.38671f);
        pos_normal_color[i * 9 + 8] = make_float3(0.6875f, 0.51953f, 0.38671f);
        //}


        meshNormal[v0] = __GEIGEN__::__add(meshNormal[v0], normal);  //normal;
        meshNormal[v1] = __GEIGEN__::__add(meshNormal[v1], normal);
        meshNormal[v2] = __GEIGEN__::__add(meshNormal[v2], normal);
    }
    for(int i = 0; i < number; i++)

    {
        int v0 = tetMesh.surface[i].x;
        int v1 = tetMesh.surface[i].y;
        int v2 = tetMesh.surface[i].z;
        //meshNormal[v0].normalize(); meshNormal[v1].normalize(); meshNormal[v2].normalize();
        pos_normal_color[i * 9 + 1] =
            make_float3(meshNormal[v0].x, meshNormal[v0].y, meshNormal[v0].z);
        pos_normal_color[i * 9 + 4] =
            make_float3(meshNormal[v1].x, meshNormal[v1].y, meshNormal[v1].z);
        pos_normal_color[i * 9 + 7] =
            make_float3(meshNormal[v2].x, meshNormal[v2].y, meshNormal[v2].z);
    }

    return pos_normal_color;
}


void draw_Scene3D()
{
    //face.mesh3Ds[0] = mesh3d;
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glTranslatef(xTrans, yTrans, zTrans);
    glRotatef(xRot, 1.0f, 0.0f, 0.0f);
    glRotatef(yRot, 0.0f, 1.0f, 0.0f);

    //draw_box3D(-2, -1, -2, 4, 4, 4, 1);
    if(drawSurface)
    {
        draw_mesh3D();
    }
    if(drawbvh)
    {
        draw_bvh();
    }

    glPopMatrix();


    glutSwapBuffers();
    //glFlush();
}
double mfsum                   = 0;
double total_time              = 0;
int    total_cg_iterations     = 0;
int    total_newton_iterations = 0;
int    start                   = -1;

void saveScreenPic(const std::string& path)
{
    std::stringstream ss;
    ss << path;
    ss.fill('0');
    ss.width(5);
    ss << step;
    std::string file_path = ss.str();

    SaveScreenShot(window_width, window_height, file_path);
}

void initFEM(tetrahedra_obj& mesh)
{

    double massSum   = 0;
    double volumeSum = 0;
    //float  angleX = FEM::PI / 4, angleY = -FEM::PI / 4, angleZ = FEM::PI / 2;
    //__GEIGEN__::Matrix3x3d rotation, rotationZ, rotationY, rotationX, eigenTest;
    //__GEIGEN__::__set_Mat_val(rotation, 1, 0, 0, 0, 1, 0, 0, 0, 1);
    //__GEIGEN__::__set_Mat_val(
    //    rotationZ, cos(angleZ), -sin(angleZ), 0, sin(angleZ), cos(angleZ), 0, 0, 0, 1);
    //__GEIGEN__::__set_Mat_val(
    //    rotationY, cos(angleY), 0, -sin(angleY), 0, 1, 0, sin(angleY), 0, cos(angleY));
    //__GEIGEN__::__set_Mat_val(
    //    rotationX, 1, 0, 0, 0, cos(angleX), -sin(angleX), 0, sin(angleX), cos(angleX));


    ipc.lengthRateLame = ipc.YoungModulus / (2 * (1 + ipc.PoissonRate));
    ipc.volumeRateLame = ipc.YoungModulus * ipc.PoissonRate
                         / ((1 + ipc.PoissonRate) * (1 - 2 * ipc.PoissonRate));
    ipc.lengthRate   = 4 * ipc.lengthRateLame / 3;
    ipc.volumeRate   = ipc.volumeRateLame + 5 * ipc.lengthRateLame / 6;
    ipc.stretchStiff = ipc.clothYoungModulus / (2 * (1 + ipc.PoissonRate));

    ipc.bendStiff = ipc.bendYoungModulus * pow(ipc.clothThickness, 3)
                    / (24 * (1 - ipc.PoissonRate * ipc.PoissonRate));

    ipc.shearStiff = 0.03 * ipc.stretchStiff * ipc.strainRate;

    printf("ipc.shearStiff: %f\n", ipc.shearStiff);


    for(int i = 0; i < mesh.tetrahedraNum; i++)
    {
        __GEIGEN__::Matrix3x3d DM;
        __calculateDms3D_double(mesh.vertexes.data(), mesh.tetrahedras[i], DM);  //calculateDms3D_double(mesh.vertexes, mesh.tetrahedras[i], 0);

        __GEIGEN__::Matrix3x3d DM_inverse;
        __GEIGEN__::__Inverse(DM, DM_inverse);

        double vlm = calculateVolum(mesh.vertexes.data(), mesh.tetrahedras[i]);

        mesh.masses[mesh.tetrahedras[i].x] += vlm * ipc.density / 4;
        mesh.masses[mesh.tetrahedras[i].y] += vlm * ipc.density / 4;
        mesh.masses[mesh.tetrahedras[i].z] += vlm * ipc.density / 4;
        mesh.masses[mesh.tetrahedras[i].w] += vlm * ipc.density / 4;

        massSum += vlm * ipc.density;
        volumeSum += vlm;
        mesh.DM_inverse.push_back(DM_inverse);
        mesh.volum.push_back(vlm);


        double lengthRateLame =
            mesh.vert_youngth_modules[i] / (2 * (1 + ipc.PoissonRate));
        double volumeRateLame = mesh.vert_youngth_modules[i] * ipc.PoissonRate
                                / ((1 + ipc.PoissonRate) * (1 - 2 * ipc.PoissonRate));
        double lengthRate = 4 * lengthRateLame / 3;
        double volumeRate = volumeRateLame + 5 * lengthRateLame / 6;

        mesh.lengthRate.push_back(lengthRate);
        mesh.volumeRate.push_back(volumeRate);
    }

    for(int i = 0; i < mesh.triangles.size(); i++)
    {
        __GEIGEN__::Matrix2x2d DM;
        __calculateDm2D_double(mesh.vertexes.data(), mesh.triangles[i], DM);

        __GEIGEN__::Matrix2x2d DM_inverse;
        __GEIGEN__::__Inverse2x2(DM, DM_inverse);

        double area = calculateArea(mesh.vertexes.data(), mesh.triangles[i]);
        area *= ipc.clothThickness;
        mesh.area.push_back(area);


        mesh.masses[mesh.triangles[i].x] += ipc.clothDensity * area / 3;
        mesh.masses[mesh.triangles[i].y] += ipc.clothDensity * area / 3;
        mesh.masses[mesh.triangles[i].z] += ipc.clothDensity * area / 3;

        massSum += area * ipc.clothDensity;
        volumeSum += area;
        mesh.tri_DM_inverse.push_back(DM_inverse);
    }

    mesh.meanMass = massSum / mesh.vertexNum;
    printf("meanMass: %f\n", mesh.meanMass);
    mesh.meanVolum = volumeSum / mesh.vertexNum;
}

void DefaultSettings()
{
    // global settings
    ipc.density        = 1e3;
    ipc.PoissonRate    = 0.49;
    //ipc.lengthRateLame = ipc.YoungModulus / (2 * (1 + ipc.PoissonRate));
    //ipc.volumeRateLame = ipc.YoungModulus * ipc.PoissonRate
    //                     / ((1 + ipc.PoissonRate) * (1 - 2 * ipc.PoissonRate));
    //ipc.lengthRate        = 4 * ipc.lengthRateLame / 3;
    //ipc.volumeRate        = ipc.volumeRateLame + 5 * ipc.lengthRateLame / 6;
    ipc.frictionRate      = 0.4;
    ipc.gd_frictionRate   = 0.4;
    ipc.clothThickness    = 1e-3;
    ipc.clothYoungModulus = 1e6;
    ipc.bendYoungModulus  = 1e5;
    //ipc.stretchStiff      = ipc.clothYoungModulus / (2 * (1 + ipc.PoissonRate));
    //ipc.shearStiff        = ipc.stretchStiff * 0.3;
    ipc.clothDensity      = 2e2;
    ipc.strainRate        = 100;
    ipc.softMotionRate    = 1e0;
    ipc.bendStiff         = 3e-4;
    ipc.Newton_solver_threshold = 1e-2;
    ipc.pcg_threshold           = 1e-4;
    ipc.IPC_dt                  = 1e-2;
    ipc.relative_dhat           = 1e-3;
    //ipc.bendStiff = ipc.bendYoungModulus * pow(ipc.clothThickness, 3)
    //                / (24 * (1 - ipc.PoissonRate * ipc.PoissonRate));
    //ipc.shearStiff = 0.03 * ipc.stretchStiff * ipc.strainRate;
}
//int  meshids = 0;
void LoadSettings()
{
    bool successfulRead = false;

    //read file
    std::ifstream infile;


    std::string DEFAULT_CONFIG_FILE = std::string{gipc::assets_dir()} + "scene/parameterSetting.txt";


    infile.open(DEFAULT_CONFIG_FILE, std::ifstream::in);
    if(successfulRead = infile.is_open())
    {
        int  tempEnum;
        char ignoreToken[256];

        // global settings:
        infile >> ignoreToken >> ipc.density;
        infile >> ignoreToken >> ipc.PoissonRate;
        infile >> ignoreToken >> ipc.frictionRate;
        infile >> ignoreToken >> ipc.gd_frictionRate;
        infile >> ignoreToken >> ipc.clothThickness;
        infile >> ignoreToken >> ipc.clothYoungModulus;
        infile >> ignoreToken >> ipc.bendYoungModulus;
        //infile >> ignoreToken >> ipc.shearStiff;
        infile >> ignoreToken >> ipc.clothDensity;
        infile >> ignoreToken >> ipc.strainRate;
        infile >> ignoreToken >> ipc.softMotionRate;
        //infile >> ignoreToken >> ipc.bendStiff;
        infile >> ignoreToken >> collision_detection_buff_scale;
        infile >> ignoreToken >> motion_rate;
        infile >> ignoreToken >> ipc.IPC_dt;
        infile >> ignoreToken >> ipc.pcg_threshold;
        infile >> ignoreToken >> ipc.Newton_solver_threshold;
        infile >> ignoreToken >> ipc.relative_dhat;
        //infile >> ignoreToken >> meshids;



        //ipc.shearStiff =
        infile.close();
    }

    if(!successfulRead)
    {
        std::cerr << "Waning: failed loading settings, set to defaults." << std::endl;
        DefaultSettings();
    }
}

void set_case1()
{
    double                    dist       = 0.2;
    int                       count      = 4;
    int                       count_Y    = 4;
    double                    fem_height = -0.8;
    double                    abd_height = -0.6;
    gipc::SimpleSceneImporter importer;

    linear_system_buff_scale = 1.0;

    double Youngth_Modulus = 1e4;
    for(int k = 0; k < count_Y; ++k)
    {
        for(int i = 0; i < count; i++)
        {
            for(int j = 0; j < count; j++)
            {

                gipc::Vector2 ij{i, j};
                gipc::Vector2 pos =
                    ij * dist - gipc::Vector2::Ones() * dist * (count - 1) / 2.0;

                double3 position_offset =
                    make_double3(-pos.x(), -abd_height - 2 * dist * k, -pos.y());
                double          scale     = 0.4;
                Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
                transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * scale;
                transform.block<3, 1>(0, 3) = -Eigen::Vector3d(
                    position_offset.x, position_offset.y, position_offset.z);

                importer.load_geometry(tetMesh,
                                       3,
                                       gipc::BodyType::ABD,
                                       transform,
                                       1e5,
                                       assets_dir + "tetMesh/cube.msh",
                                       ipc.pcg_data.P_type);
            }
        }
    }

    for(int k = 0; k < count_Y; ++k)
    {
        for(int i = 0; i < count; i++)
        {
            for(int j = 0; j < count; j++)
            {

                gipc::Vector2 ij{i, j};
                gipc::Vector2 pos =
                    ij * dist - gipc::Vector2::Ones() * dist * (count - 1) / 2.0;

                double3 position_offset =
                    make_double3(-pos.x(), -fem_height - 2 * dist * k, -pos.y());
                double          scale     = 0.4;
                Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
                transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * scale;
                transform.block<3, 1>(0, 3) = -Eigen::Vector3d(
                    position_offset.x, position_offset.y, position_offset.z);

                importer.load_geometry(tetMesh,
                                       3,
                                       gipc::BodyType::FEM,
                                       transform,
                                       Youngth_Modulus,
                                       assets_dir + "tetMesh/cube.msh",
                                       ipc.pcg_data.P_type);
            }
        }
    }
}


void set_case2()
{
    ipc.pcg_data.P_type = 1;
    gipc::SimpleSceneImporter importer;
    double                    scale           = 0.2;
    double3                   position_offset = make_double3(0, -0.5, 0);
    Eigen::Matrix4d           transform       = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * scale;
    transform.block<3, 1>(0, 3) =
        -Eigen::Vector3d(position_offset.x, position_offset.y, position_offset.z);

    linear_system_buff_scale = 1.0;
    double Youngth_Modulus = 1e4;
    std::string mesh0_path      = assets_dir + "tetMesh/bunny2.msh";
    importer.load_geometry(tetMesh,
                           3,
                           gipc::BodyType::ABD,
                           transform,
                           Youngth_Modulus,
                           mesh0_path,
                           ipc.pcg_data.P_type);

    position_offset             = make_double3(0, 0.65, 0);
    transform                   = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * scale;
    transform.block<3, 1>(0, 3) =
        -Eigen::Vector3d(position_offset.x, position_offset.y, position_offset.z);

    std::string mesh1_path = mesh0_path;
    importer.load_geometry(tetMesh,
                           3,
                           gipc::BodyType::FEM,
                           transform,
                           Youngth_Modulus,
                           mesh1_path,
                           ipc.pcg_data.P_type);


    position_offset             = make_double3(0, 0, 0);
    transform                   = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * 1.0;
    transform.block<3, 1>(0, 3) =
        -Eigen::Vector3d(position_offset.x, position_offset.y, position_offset.z);
    std::string mesh2_path = assets_dir + "triMesh/cloth_high.obj";

    importer.load_geometry(tetMesh,
                           2,
                           gipc::BodyType::FEM,
                           transform,
                           1e4,
                           mesh2_path,
                           ipc.pcg_data.P_type);
}

void set_case3()
{

    gipc::SimpleSceneImporter importer{assets_dir + "scene/json/wrecking-ball-simple.json",
                                       assets_dir + "tetMesh/wrecking-ball-mesh/",
                                       gipc::BodyType::ABD};
    linear_system_buff_scale = 1.0;
    importer.import_scene(tetMesh);
}

void set_case4()
{
    ipc.pcg_data.P_type = 0;
    linear_system_buff_scale = 1.0;
    gipc::SimpleSceneImporter importer;
    double                    scale           = 0.6;
    double3                   position_offset = make_double3(0, 1.0, 0);

    using Transform = Eigen::Transform<double, 3, Eigen::Affine>;
    Transform t     = Transform::Identity();
    t.translate(Eigen::Vector3d{0, 1.0, 0});
    t.scale(scale);
    t.rotate(Eigen::AngleAxisd(3.1415926 / 2, Eigen::Vector3d::UnitX()));
    Eigen::Matrix4d transform = t.matrix();

    std::string mesh_path = assets_dir + "triMesh/cloth_high.obj";
    importer.load_geometry(tetMesh,
                           2,
                           gipc::BodyType::FEM,
                           transform,
                           1e4,
                           mesh_path,
                           ipc.pcg_data.P_type);

    int          fixed_vertex_num = 0;
    const double eps              = 1e-4;
    double       max_y            = tetMesh.maxTConer.y;
    double       min_x            = tetMesh.minTConer.x;
    double       max_x            = tetMesh.maxTConer.x;
    for(int i = 0; i < tetMesh.vertexNum; i++)
    {
        if(tetMesh.vertexes[i].y > max_y - eps
           && (tetMesh.vertexes[i].x < min_x + eps || tetMesh.vertexes[i].x > max_x - eps))
        {
            tetMesh.boundaryTypies[i] = 1;
            fixed_vertex_num++;
        }
    }
    std::cout << "fixed vertex num: " << fixed_vertex_num << std::endl;
}

void set_case5()
{
    ipc.pcg_data.P_type = 1;
    linear_system_buff_scale = 2.0;
    gipc::SimpleSceneImporter importer;
    double                    scale = 1.0;
    Eigen::Vector3d           position_offset{0, 0, 0};

    using Transform = Eigen::Transform<double, 3, Eigen::Affine>;
    Transform t     = Transform::Identity();
    t.translate(position_offset);
    t.scale(scale);
    Eigen::Matrix4d transform = t.matrix();

    std::string mesh_path       = assets_dir + "tetMesh/high_mat.msh";
    double Youngth_Modulus = 1e4;
    ipc.PoissonRate        = 0.48;
    importer.load_geometry(tetMesh,
                           3,
                           gipc::BodyType::FEM,
                           transform,
                           Youngth_Modulus,
                           mesh_path,
                           ipc.pcg_data.P_type);

    // no gravity
    for(int i = 0; i < tetMesh.vertexes.size(); i++)
    {
        tetMesh.apply_gravity[i] = 0;
    }

    const double eps = 1e-4;
    for(int i = 0; i < tetMesh.vertexNum; i++)
    {
        if(tetMesh.vertexes[i].x < -0.5 + eps || tetMesh.vertexes[i].x > 0.5 - eps)
        {
            tetMesh.targetIndex.push_back(i);
            tetMesh.targetPos.push_back(tetMesh.vertexes[i]);
        }
    }
    tetMesh.softNum = tetMesh.targetIndex.size();
    std::cout << "soft constraint num: " << tetMesh.softNum << std::endl;
    ipc.softMotionRate = 1;

    const double angular_vel = 3.14159265358979323846/5;
    d_tetMesh.update_soft_constraint_functor =
        [angular_vel](double3 vertex, int step_id, double ipc_dt) -> double3
    {
        double3 rotated_vertex = vertex;
        if(vertex.x < 0)
        {
            // rotate along x axis clockwise
            rotated_vertex = {vertex.x,
                              vertex.y * std::cos(angular_vel * ipc_dt)
                                  - vertex.z * std::sin(angular_vel * ipc_dt),
                              vertex.y * std::sin(angular_vel * ipc_dt)
                                  + vertex.z * std::cos(angular_vel * ipc_dt)};
        }
        if(vertex.x > 0)
        {
            // rotate along x axis counterclockwise
            rotated_vertex = {vertex.x,
                              vertex.y * std::cos(-angular_vel * ipc_dt)
                                  - vertex.z * std::sin(-angular_vel * ipc_dt),
                              vertex.y * std::sin(-angular_vel * ipc_dt)
                                  + vertex.z * std::cos(-angular_vel * ipc_dt)};
        }
        return rotated_vertex;
    };
}

void set_case6()
{
    linear_system_buff_scale = 2.0;
    ipc.pcg_data.P_type = 1;
    double scale      = 0.3;
    double dist       = scale / 2;
    int    count      = 8;
    int    count_Y    = 15;
    double fem_height = global_offset + 1 - 0.8;
    double abd_height = fem_height - dist;


    for(int k = 0; k < count_Y; ++k)
    {
        for(int i = 0; i < count; i++)
        {
            for(int j = 0; j < count; j++)
            {

                gipc::Vector2 ij{i, j};
                gipc::Vector2 pos =
                    ij * dist - gipc::Vector2::Ones() * dist * (count - 1) / 2.0;


                double3 position_offset =
                    double3{-pos.x(), -abd_height - 2 * dist * k, -pos.y()};
                Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
                transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * scale;
                transform.block<3, 1>(0, 3) = -Eigen::Vector3d(
                    position_offset.x, position_offset.y, position_offset.z);
                tetMesh.load_tetrahedraMesh(assets_dir + "tetMesh/cube.msh",
                                            transform,
                                            1e6,
                                            gipc::BodyType::ABD);
            }
        }
    }

    for(int k = 0; k < count_Y; ++k)
    {
        for(int i = 0; i < count; i++)
        {
            for(int j = 0; j < count; j++)
            {
                gipc::Vector2 ij{i, j};
                gipc::Vector2 pos =
                    ij * dist - gipc::Vector2::Ones() * dist * (count - 1) / 2.0;

                double3 position_offset =
                    double3{-pos.x(), -fem_height - 2 * dist * k, -pos.y()};
                Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
                transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * scale;
                transform.block<3, 1>(0, 3) = -Eigen::Vector3d(
                    position_offset.x, position_offset.y, position_offset.z);
                tetMesh.load_tetrahedraMesh(assets_dir + "tetMesh/cube.msh",
                                            transform,
                                            5e4,
                                            gipc::BodyType::ABD);
            }
        }
    }
    
    gipc::SimpleSceneImporter importer;
    using Transform = Eigen::Transform<double, 3, Eigen::Affine>;
    Transform t     = Transform::Identity();
    t.scale(1.5);
    t.translate(Eigen::Vector3d(0, 0.35, 0));
    std::string mesh_path = assets_dir + "triMesh/cloth_high.obj";
    importer.load_geometry(tetMesh,
                           2,
                           gipc::BodyType::FEM,
                           t.matrix(),
                           1e4,
                           mesh_path,
                           ipc.pcg_data.P_type);

    const double eps = 1e-4;
    for(int i = 0; i < tetMesh.vertexNum; i++)
    {
        if(tetMesh.vertexes[i].x < -1.5 + eps || tetMesh.vertexes[i].x > 1.5 - eps)
        {
            tetMesh.boundaryTypies[i] = 1;
        }
    }

    ipc.relative_dhat = 1e-3;
    ipc.strainRate    = 1e6;
}

void setMAS_partition()
{
    tetMesh.partId_map_real.resize(tetMesh.part_offset * BANKSIZE, -1);
    tetMesh.real_map_partId.resize(tetMesh.partId.size());
    int index = 0;
    for(int i = 0; i < tetMesh.partId.size(); i++)
    {
        tetMesh.partId_map_real[BANKSIZE * tetMesh.partId[i] + index] = i;
        index++;
        if(i <= tetMesh.partId.size() - 2)
        {
            if(tetMesh.partId[i + 1] != tetMesh.partId[i])
            {
                index = 0;
            }
        }
    }
    index = 0;
    for(int i = 0; i < tetMesh.partId_map_real.size(); i++)
    {

        if(tetMesh.partId_map_real[i] == index)
        {
            tetMesh.real_map_partId[index] = i;
            index++;
        }
    }
}

#include "benchmark_cloth_v1.inc"

void initScene()
{
    std::filesystem::exists(metis_dir) || std::filesystem::create_directory(metis_dir);
    ipc.pcg_data.P_type = 1;

    int scene_no = gipc::benchmark::options.scene;
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!ABD must be loaded before FEM!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if(std::getenv("CIPC_SCENE"))
        set_case8();
    else switch(scene_no)
    {
        case 0:  // box pipe
            set_case1();
            break;
        case 1:  // soft-rigid-cloth coupling
            set_case2();
            break;
        case 2:  //wrecking ball case
            set_case3();
            break;
        case 3:  //fixed cloth
            set_case4();
            break;
        case 4:  //twisting mat
            set_case5();
            break;
        case 5:  //box pipe large scale and cloth
            set_case6();
            break;
    }


    // Apply after scene-specific defaults, before creating the PCG solver.
    ipc.Newton_solver_threshold *= gipc::benchmark::options.newton_tolerance_scale;
    ipc.pcg_threshold *= gipc::benchmark::options.pcg_tolerance_scale;
    setMAS_partition();


    tetMesh.getSurface();

    initFEM(tetMesh);
    //device_TetraData d_tetMesh;
    d_tetMesh.Malloc_DEVICE_MEM(tetMesh.vertexNum,
                                tetMesh.tetrahedraNum,
                                tetMesh.triangleNum,
                                tetMesh.softNum,
                                tetMesh.tri_edges.size(),
                                tetMesh.abd_fem_count_info.total_body_num());

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.masses,
                              tetMesh.masses.data(),
                              tetMesh.vertexNum * sizeof(double),
                              cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.apply_gravity,
                              tetMesh.apply_gravity.data(),
                              tetMesh.vertexNum * sizeof(int),
                              cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.lengthRate,
                              tetMesh.lengthRate.data(),
                              tetMesh.tetrahedraNum * sizeof(double),
                              cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.volumeRate,
                              tetMesh.volumeRate.data(),
                              tetMesh.tetrahedraNum * sizeof(double),
                              cudaMemcpyHostToDevice));


    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.volum,
                              tetMesh.volum.data(),
                              tetMesh.tetrahedraNum * sizeof(double),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.vertexes,
                              tetMesh.vertexes.data(),
                              tetMesh.vertexNum * sizeof(double3),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.o_vertexes,
                              tetMesh.vertexes.data(),
                              tetMesh.vertexNum * sizeof(double3),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.tetrahedras,
                              tetMesh.tetrahedras.data(),
                              tetMesh.tetrahedraNum * sizeof(uint4),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.DmInverses,
                              tetMesh.DM_inverse.data(),
                              tetMesh.tetrahedraNum * sizeof(__GEIGEN__::Matrix3x3d),
                              cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.BoundaryType,
                              tetMesh.boundaryTypies.data(),
                              tetMesh.vertexNum * sizeof(int),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.velocities,
                              tetMesh.velocities.data(),
                              tetMesh.vertexNum * sizeof(double3),
                              cudaMemcpyHostToDevice));


    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.targetIndex,
                              tetMesh.targetIndex.data(),
                              tetMesh.softNum * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.targetVert,
                              tetMesh.targetPos.data(),
                              tetMesh.softNum * sizeof(double3),
                              cudaMemcpyHostToDevice));

    d_tetMesh.host_target_indices  = tetMesh.targetIndex;
    d_tetMesh.host_target_vertices = tetMesh.targetPos;

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.triDmInverses,
                              tetMesh.tri_DM_inverse.data(),
                              tetMesh.triangleNum * sizeof(__GEIGEN__::Matrix2x2d),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.area,
                              tetMesh.area.data(),
                              tetMesh.triangleNum * sizeof(double),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.triangles,
                              tetMesh.triangles.data(),
                              tetMesh.triangleNum * sizeof(uint3),
                              cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.tri_edges,
                              tetMesh.tri_edges.data(),
                              tetMesh.tri_edges.size() * sizeof(uint2),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.tri_edge_adj_vertex,
                              tetMesh.tri_edges_adj_points.data(),
                              tetMesh.tri_edges.size() * sizeof(uint2),
                              cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.body_id_to_boundary_type,
                              tetMesh.body_id_to_is_fixed.data(),
                              tetMesh.body_id_to_is_fixed.size() * sizeof(int),
                              cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.point_id_to_body_id,
                              tetMesh.point_id_to_body_id.data(),
                              tetMesh.point_id_to_body_id.size() * sizeof(int),
                              cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.tet_id_to_body_id,
                              tetMesh.tet_id_to_body_id.data(),
                              tetMesh.tet_id_to_body_id.size() * sizeof(int),
                              cudaMemcpyHostToDevice));


    printf("stretchStiff:  %f,  shearStiff:   %f\n", ipc.stretchStiff, ipc.shearStiff);

    ipc.vertexNum      = tetMesh.vertexNum;
    ipc.tetrahedraNum  = tetMesh.tetrahedraNum;
    ipc._vertexes      = d_tetMesh.vertexes;
    ipc._rest_vertexes = d_tetMesh.rest_vertexes;
    ipc.surf_vertexNum = tetMesh.surfVerts.size();
    ipc.surface_Num    = tetMesh.surface.size();
    ipc.edge_Num       = tetMesh.surfEdges.size();
    ipc.tri_edge_num   = tetMesh.tri_edges.size();

    //ipc.IPC_dt = 0.01 / 1.0;//1.0 / 30;//1.0 / 100;
    ipc.MAX_CCD_COLLITION_PAIRS_NUM =
        1 * collision_detection_buff_scale
        * (((double)(ipc.surface_Num * 15 + ipc.edge_Num * 10))
           * std::max((ipc.IPC_dt / 0.01), 2.0));
    ipc.MAX_COLLITION_PAIRS_NUM = (ipc.surf_vertexNum * 3 + ipc.edge_Num * 2)
                                  * 3 * collision_detection_buff_scale;

    ipc.triangleNum        = tetMesh.triangleNum;
    ipc.targetVert         = d_tetMesh.targetVert;
    ipc.targetInd          = d_tetMesh.targetIndex;
    ipc.softNum            = tetMesh.softNum;
    ipc.abd_fem_count_info = tetMesh.abd_fem_count_info;

    std::cout << "ABD FEM count info: \n"
              << ipc.abd_fem_count_info << std::endl;


    printf("vertNum: %d      tetraNum: %d      faceNum: %d\n",
           ipc.vertexNum,
           ipc.tetrahedraNum,
           ipc.surface_Num);
    printf("surfVertNum: %d      surfEdgesNum: %d\n", ipc.surf_vertexNum, ipc.edge_Num);
    printf("maxCollisionPairsNum_CCD: %d      maxCollisionPairsNum: %d\n",
           ipc.MAX_CCD_COLLITION_PAIRS_NUM,
           ipc.MAX_COLLITION_PAIRS_NUM);

    //ipc.USE_MAS = false;
    ipc.MALLOC_DEVICE_MEM();

    CUDA_SAFE_CALL(cudaMemcpy(
        ipc._faces, tetMesh.surface.data(), ipc.surface_Num * sizeof(uint3), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(
        ipc._edges, tetMesh.surfEdges.data(), ipc.edge_Num * sizeof(uint2), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(ipc._surfVerts,
                              tetMesh.surfVerts.data(),
                              ipc.surf_vertexNum * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
    ipc.initBVH(d_tetMesh.BoundaryType, d_tetMesh.point_id_to_body_id);

    if(ipc.pcg_data.P_type && true)
    {
        int neighborListSize = tetMesh.getVertNeighbors();
        ipc.pcg_data.MP.initPreconditioner_Neighbor(ipc.vertexNum - tetMesh.abd_vertexOffset,
                                                    tetMesh.abd_vertexOffset,
                                                    neighborListSize,
                                                    ipc._collisonPairs,
                                                    tetMesh.part_offset * BANKSIZE);

        ipc.pcg_data.MP.neighborListSize = neighborListSize;
        CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_neighborListInit,
                                  tetMesh.neighborList.data(),
                                  neighborListSize * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_neighborStart,
                                  tetMesh.neighborStart.data(),
                                  (ipc.vertexNum - tetMesh.abd_vertexOffset) * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_neighborNumInit,
                                  tetMesh.neighborNum.data(),
                                  (ipc.vertexNum - tetMesh.abd_vertexOffset) * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_partId_map_real,
                                  tetMesh.partId_map_real.data(),
                                  tetMesh.part_offset * BANKSIZE * sizeof(int),
                                  cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_real_map_partId,
                                  tetMesh.real_map_partId.data(),
                                  tetMesh.real_map_partId.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));

        ipc.pcg_data.MP.initPreconditioner_Matrix();
    }

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.rest_vertexes,
                              d_tetMesh.o_vertexes,
                              ipc.vertexNum * sizeof(double3),
                              cudaMemcpyDeviceToDevice));


#ifdef USE_QUADRATIC_BENDING
    // Precompute Q matrices for quadratic bending
    if(tetMesh.tri_edges.size() > 0)
    {
        printf("Precomputing Q matrices for quadratic bending (%zu edges)...\n",
               tetMesh.tri_edges.size());

        // Allocate host memory for Q matrices
        std::vector<Eigen::Matrix4d> Q_host(tetMesh.tri_edges.size());

        // Download rest positions, edges, and adjacency from device
        std::vector<double3> rest_verts_host(ipc.vertexNum);
        CUDA_SAFE_CALL(cudaMemcpy(rest_verts_host.data(),
                                  d_tetMesh.rest_vertexes,
                                  ipc.vertexNum * sizeof(double3),
                                  cudaMemcpyDeviceToHost));

        // Call precomputation function (defined in femEnergy.cu)
        PrepareQuadBendingQ(rest_verts_host.data(),
                            tetMesh.tri_edges.data(),
                            tetMesh.tri_edges_adj_points.data(),  // CPU端叫tri_edges_adj_points
                            tetMesh.tri_edges.size(),
                            Q_host.data());

        // Upload Q matrices to device
        CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.quad_bending_Q,
                                  Q_host.data(),
                                  tetMesh.tri_edges.size() * sizeof(Eigen::Matrix4d),
                                  cudaMemcpyHostToDevice));

        printf("Quadratic bending Q matrices uploaded successfully.\n");

        // Optional: Print first Q matrix for verification
        if(tetMesh.tri_edges.size() > 0)
        {
            printf("First Q matrix:\n");
            for(int i = 0; i < 4; i++)
            {
                printf("  [%10.6f %10.6f %10.6f %10.6f]\n",
                       Q_host[0](i, 0),
                       Q_host[0](i, 1),
                       Q_host[0](i, 2),
                       Q_host[0](i, 3));
            }

            // Check for NaN or Inf
            int nan_count = 0;
            int inf_count = 0;
            for(size_t i = 0; i < Q_host.size(); i++)
            {
                for(int r = 0; r < 4; r++)
                {
                    for(int c = 0; c < 4; c++)
                    {
                        if(std::isnan(Q_host[i](r, c)))
                            nan_count++;
                        if(std::isinf(Q_host[i](r, c)))
                            inf_count++;
                    }
                }
            }
            if(nan_count > 0 || inf_count > 0)
            {
                printf("WARNING: Q matrices contain %d NaN values and %d Inf values!\n",
                       nan_count,
                       inf_count);
            }
        }
    }
#endif


    ipc.buildBVH();
    ipc.init(tetMesh.meanMass, tetMesh.meanVolum, tetMesh.minConer, tetMesh.maxConer, linear_system_buff_scale);

    printf("bboxDiagSize2: %f\n", ipc.bboxDiagSize2);
    printf("maxConer: %f  %f   %f           minCorner: %f  %f   %f\n",
           tetMesh.maxConer.x,
           tetMesh.maxConer.y,
           tetMesh.maxConer.z,
           tetMesh.minConer.x,
           tetMesh.minConer.y,
           tetMesh.minConer.z);

    printf("restSNKE: %f\n", ipc.RestNHEnergy);
    ipc.buildCP();

    ipc._moveDir          = ipc.pcg_data.dx;
    ipc.animation_subRate = 1.0 / motion_rate;
    //ipc.animation_fullRate = ipc.animation_subRate;
    ipc.computeXTilta(d_tetMesh, 1);
    ///////////////////////////////////////////////////////////////////////////////////

    ipc.create_LinearSystem(d_tetMesh);

    bvs.resize(2 * ipc.edge_Num - 1);
    nodes.resize(2 * ipc.edge_Num - 1);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(
        &bvs[0], ipc.bvh_e._bvs, (2 * ipc.edge_Num - 1) * sizeof(AABB), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(
        &nodes[0], ipc.bvh_e._nodes, (2 * ipc.edge_Num - 1) * sizeof(Node), cudaMemcpyDeviceToHost));
}


void outputAnimationMeshInfo(std::string pathCloth, std::string pathBody)
{
    std::stringstream ss;
    ss << pathCloth;
    ss.fill('0');
    ss.width(5);
    ss << (surfNumId) / 1;  // / 10;
    //if (surfNumId % 10 != 0) return;
    ss << ".obj";
    std::string file_path = ss.str();
    std::ofstream    outSurf(file_path);

    std::map<int, int> meshToSurf;
    for(int i = 0; i < bodyVertOffset; i++)
    {
        const auto& pos = tetMesh.vertexes[i];
        outSurf << "v " << pos.x << " " << pos.y << " " << pos.z << std::endl;
        //meshToSurf[tetMesh.surfVerts[i]] = i;
    }

    for(int i = 0; i < tetMesh.triangles.size(); i++)
    {
        const auto& tri = tetMesh.triangles[i];
        outSurf << "f " << tri.x + 1 << " " << tri.y + 1 << " " << tri.z + 1 << std::endl;
    }
    outSurf.close();

    std::stringstream ss2;
    ss2 << pathBody;
    ss2.fill('0');
    ss2.width(5);
    ss2 << (surfNumId) / 1;  // / 10;
    //if (surfNumId % 10 != 0) return;
    ss2 << ".obj";
    std::string file_path2 = ss2.str();
    std::ofstream    outSurf2(file_path2);

    //std::map<int, int> meshToSurf;
    for(int i = bodyVertOffset; i < tetMesh.vertexes.size(); i++)
    {
        const auto& pos = tetMesh.vertexes[i];
        outSurf2 << "v " << pos.x << " " << pos.y << " " << pos.z << std::endl;
        //meshToSurf[tetMesh.surfVerts[i]] = i;
    }

    for(int i = 0; i < clothFaceOffset; i++)
    {
        const auto& tri = tetMesh.surface[i];
        outSurf2 << "f " << tri.x + 1 - bodyVertOffset << " " << tri.y + 1 - bodyVertOffset
                 << " " << tri.z + 1 - bodyVertOffset << std::endl;
    }
    outSurf2.close();
    surfNumId++;
}
bool pri = true;
void display(void)
{
    draw_Scene3D();
    std::filesystem::exists(std::string{gipc::output_dir()})
        || std::filesystem::create_directory(std::string{gipc::output_dir()});
    auto output_path = std::string{gipc::output_dir()} + "saveSurface/";

    std::filesystem::exists(output_path) || std::filesystem::create_directory(output_path);

    //if(stop)
    //    return;


    ipc.IPC_Solver(d_tetMesh);

    if(ipc.animation && true)
    {
        std::string filename =
            "triMesh/body4/postcvpr_big_body_" + std::to_string(frameId + 1) + ".obj";
        frameId++;
        tetMesh.load_animation(filename, 1, make_double3(-1, -0.5, -0.5));
        CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.targetVert,
                                  tetMesh.targetPos.data(),
                                  tetMesh.softNum * sizeof(double3),
                                  cudaMemcpyHostToDevice));
    }


    CUDA_SAFE_CALL(cudaMemcpy(
        &bvs[0], ipc.bvh_e._bvs, (2 * ipc.edge_Num - 1) * sizeof(AABB), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(
        &nodes[0], ipc.bvh_e._nodes, (2 * ipc.edge_Num - 1) * sizeof(Node), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(tetMesh.vertexes.data(),
                              ipc._vertexes,
                              ipc.vertexNum * sizeof(double3),
                              cudaMemcpyDeviceToHost));


    if(screenshot)
    {
        std::stringstream ss;
        ss << "saveScreen/step_";
        ss.fill('0');
        ss.width(5);
        ss << step / 1;
        std::string file_path = ss.str();
        SaveScreenShot(window_width, window_height, file_path);
    }
    step++;
    printf("step:  %d\n", step);

    //if(step >= 160)
    //{
    //    std::cout << "step: " << step << " finished." << std::endl;
    //    exit(0);
    //}
}

void init(void)
{
    Init_CUDA();

    //main2();

    GLenum err = glewInit();
    if(GLEW_OK != err)
    {
        /* Problem: glewInit failed, something is seriously wrong. */
        std::cerr << "Error: " << glewGetErrorString(err) << std::endl;
    }
    std::cerr << "Status: Using GLEW " << glewGetString(GLEW_VERSION) << std::endl;
    glClearColor(0.0, 0.0, 0.0, 1.0);


    LoadSettings();

    ipc.build_gipc_system(d_tetMesh);

    initScene();

    if(!isSetShader)
    {
        glViewport(0, 0, window_width, window_height);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(45.0, (float)window_width / window_height, 10.1f, 500.0);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glTranslatef(0.0f, 0.0f, -3.0f);
    }
    else
    {
        glGenBuffers(1, &PN_vbo_);
        glGenVertexArrays(1, &VAO);
    }
    //glEnable(GL_DEPTH_TEST);
}


void idle_func()
{
    glutPostRedisplay();
}

void reshape_func(GLint width, GLint height)
{
    //window_width = width;
    //window_height = height;

    glViewport(0, 0, width, height);
    if(!isSetShader)
    {
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();

        gluPerspective(45.0, (float)width / height, 0.1, 500.0);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glTranslatef(0.0f, 0.0f, -3.0f);
    }
    //glTranslatef(0.5f, 0.5f, -4.0f);
}

void keyboard_func(unsigned char key, int x, int y)
{
    if(key == 'w')
    {
        zTrans += .3f;
    }

    if(key == 's')
    {
        zTrans -= .3f;
    }

    if(key == 'a')
    {
        xTrans += .3f;
    }

    if(key == 'd')
    {
        xTrans -= .3f;
    }

    if(key == 'q')
    {
        yTrans -= .3f;
    }

    if(key == 'e')
    {
        yTrans += .3f;
    }

    if(key == '/')
    {
        screenshot = !screenshot;
    }

    if(key == '9')
    {
        saveSurface = !saveSurface;
    }

    if(key == 'k')
    {
        drawSurface = !drawSurface;
    }

    if(key == 'f')
    {
        drawbvh = !drawbvh;
    }

    if(key == ' ')
    {
        stop = !stop;
    }
    glutPostRedisplay();
}

void special_keyboard_func(int key, int x, int y)
{
    glutPostRedisplay();
}

void mouse_func(int button, int state, int x, int y)
{
    if(state == GLUT_DOWN)
    {
        buttonState = 1;
    }
    else if(state == GLUT_UP)
    {
        buttonState = 0;
    }

    ox = x;
    oy = y;

    glutPostRedisplay();
}

void motion_func(int x, int y)
{
    float dx, dy;
    dx = (float)(x - ox);
    dy = (float)(y - oy);

    if(buttonState == 1)
    {
        xRot += dy / 5.0f;
        yRot += dx / 5.0f;
    }

    ox = x;
    oy = y;

    glutPostRedisplay();
}


void SpecialKey(GLint key, GLint x, GLint y)
{
    if(key == GLUT_KEY_DOWN)
    {
        change = true;
        initPath -= 1;
        if(initPath < 0)
        {
            initPath = obj_pathes.size() - 1;
        }
    }

    if(key == GLUT_KEY_UP)
    {
        change = true;
        initPath += 1;
        if(initPath == obj_pathes.size())
        {
            initPath = 0;
        }
    }
    glutPostRedisplay();
}


int run_benchmark()
{
    namespace bench = gipc::benchmark;
    const std::filesystem::path output(bench::options.output_dir);
    if(std::filesystem::exists(output))
        throw std::runtime_error("Benchmark output already exists");
    std::filesystem::create_directories(output);
    // Same physics initialization as init(), without OpenGL setup.
    Init_CUDA();
    LoadSettings();
    ipc.build_gipc_system(d_tetMesh);
    initScene();
    if(!bench::options.checkpoint_in.empty())
        ipc.load_checkpoint(d_tetMesh, bench::options.checkpoint_in);
    if(!bench::options.checkpoint_roundtrip_out.empty())
        ipc.save_checkpoint(d_tetMesh, bench::options.checkpoint_roundtrip_out);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    int device = 0, runtime = 0, driver = 0;
    cudaDeviceProp properties{};
    CUDA_SAFE_CALL(cudaGetDevice(&device));
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&properties, device));
    CUDA_SAFE_CALL(cudaRuntimeGetVersion(&runtime));
    CUDA_SAFE_CALL(cudaDriverGetVersion(&driver));
    int profile_sync_calibration_calls = 0;
    double profile_sync_calibration_ms = 0;
    if(bench::options.profile_stages)
    {
        profile_sync_calibration_calls = 100;
        const auto begin = bench::Clock::now();
        for(int i = 0; i < profile_sync_calibration_calls; ++i)
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
        profile_sync_calibration_ms = bench::elapsed_ms(begin);
    }
    gipc::Json metadata{
        {"base_commit", "c49982fac7767225c7775683734c89eb74081142"},
        {"method", bench::options.dual_graph ? "pcg_dual_graph" : "base"},
        {"scene_no", bench::options.scene}, {"frames", bench::options.frames},
        {"cloth_scene", cipc_case},
        {"cloth_baseline", cipc_case.empty() ? "" : "cloth-v1-20260911"},
        {"vertex_count", ipc.vertexNum}, {"tetrahedron_count", tetMesh.tetrahedraNum},
        {"dt", ipc.IPC_dt}, {"pcg_tolerance", ipc.pcg_threshold},
        {"newton_tolerance", ipc.Newton_solver_threshold},
        {"gpu_index", device}, {"gpu_name", properties.name},
        {"cuda_runtime", runtime}, {"cuda_driver", driver},
        {"audit_frame", bench::options.audit_frame},
        {"termination", bench::options.termination},
        {"toi_min_updates", bench::options.toi_min_updates},
        {"toi_threshold", bench::options.toi_threshold},
        {"residual_floor", bench::options.residual_floor},
        {"residual_threshold", bench::options.residual_threshold},
        {"residual_weight", bench::options.toi_threshold / bench::options.residual_threshold},
        {"profile_stages", bench::options.profile_stages},
        {"profile_detail_schema", "b1-exclusive-v1"},
        {"profile_sync_calibration_calls", profile_sync_calibration_calls},
        {"profile_sync_calibration_ms", profile_sync_calibration_ms},
        {"trace_newton", bench::options.trace_newton},
        {"diagnostic_max_ms", bench::options.diagnostic_max_ms},
        {"diagnostic_max_newton", bench::options.diagnostic_max_newton},
        {"production_measurement", bench::options.production_measurement},
        {"converter_indexed_reduce", bench::options.converter_indexed_reduce},
        {"cloth_offset_y", bench::options.cloth_offset_y},
        {"cloth_initial_vy", std::getenv("CIPC_INITIAL_VY") ? std::stod(std::getenv("CIPC_INITIAL_VY")) : 0},
        {"residual_space", "native unified ABD affine DOFs and free FEM Cartesian DOFs; original RHS scaling"},
        {"fem_point_offset", ipc.abd_fem_count_info.fem_point_offset},
        {"fem_point_count", ipc.abd_fem_count_info.fem_point_num},
        {"newton_tolerance_scale", bench::options.newton_tolerance_scale},
        {"pcg_tolerance_scale", bench::options.pcg_tolerance_scale},
        {"checkpoint_in", bench::options.checkpoint_in},
        {"checkpoint_out", bench::options.checkpoint_out},
        {"checkpoint_frame", bench::options.checkpoint_frame},
        {"checkpoint_output_root", bench::options.checkpoint_output_root},
        {"checkpoint_frames", bench::options.checkpoint_frames},
        {"initial_physical_frame", ipc.physical_frame()},
        {"timing", "synchronized wall clock; excludes exports/rendering/legacy reports; includes graph preparation"}};
    std::ofstream(output / "metadata.json") << metadata.dump(2);
    std::vector<double3> vertices(ipc.vertexNum);
    CUDA_SAFE_CALL(cudaMemcpy(vertices.data(), ipc._vertexes,
                              vertices.size() * sizeof(double3), cudaMemcpyDeviceToHost));
    auto write_vertices = [&](const std::filesystem::path& file)
    {
        std::ofstream stream(file, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(vertices.data()),
                     vertices.size() * sizeof(double3));
        if(!stream)
            throw std::runtime_error("State export failed");
    };
    write_vertices(output / "initial.f64");
    // Same complete initialization is used by each independent contact probe;
    // preserve additional persistent physical inputs for equality auditing.
    std::vector<double3> initial_velocity(ipc.vertexNum);
    std::vector<double> initial_mass(ipc.vertexNum);
    CUDA_SAFE_CALL(cudaMemcpy(initial_velocity.data(), d_tetMesh.velocities,
        initial_velocity.size()*sizeof(double3), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(initial_mass.data(), d_tetMesh.masses,
        initial_mass.size()*sizeof(double), cudaMemcpyDeviceToHost));
    std::ofstream initial_velocity_file(output / "initial_velocities.f64", std::ios::binary);
    initial_velocity_file.write(reinterpret_cast<const char*>(initial_velocity.data()), initial_velocity.size()*sizeof(double3));
    std::ofstream initial_mass_file(output / "masses.f64", std::ios::binary);
    initial_mass_file.write(reinterpret_cast<const char*>(initial_mass.data()), initial_mass.size()*sizeof(double));
    std::vector<uint4> tets(tetMesh.tetrahedraNum);
    std::vector<int> boundary(ipc.vertexNum);
    if(!tets.empty())
        CUDA_SAFE_CALL(cudaMemcpy(tets.data(), d_tetMesh.tetrahedras,
            tets.size() * sizeof(uint4), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(boundary.data(), d_tetMesh.BoundaryType,
        boundary.size() * sizeof(int), cudaMemcpyDeviceToHost));
    std::ofstream topology(output / "tetrahedra.u32", std::ios::binary);
    topology.write(reinterpret_cast<const char*>(tets.data()), tets.size() * sizeof(uint4));
    std::ofstream boundary_file(output / "boundary.i32", std::ios::binary);
    boundary_file.write(reinterpret_cast<const char*>(boundary.data()), boundary.size() * sizeof(int));
    std::vector<double3> velocities(ipc.vertexNum);
    std::ofstream states;
    std::ofstream velocity_states;
    if(bench::options.export_states)
    {
        states.open(output / "states.f64", std::ios::binary);
        velocity_states.open(output / "velocities.f64", std::ios::binary);
    }
    std::ofstream frames(output / "frames.csv");
    std::ofstream traces(output / "pcg_iterations.jsonl");
    std::ofstream termination_trace;
    if(bench::options.trace_termination)
        termination_trace.open(output / "termination_trace.jsonl");
    std::ofstream newton_trace;
    if(bench::options.trace_newton)
        newton_trace.open(output / "newton_trace.jsonl");
    std::ofstream stage_profile;
    if(bench::options.profile_stages)
        stage_profile.open(output / "stage_profile.jsonl");
    frames << "frame,newton_iterations,newton_cap_hits,pcg_solves,pcg_iterations,"
              "newton_ms,pcg_ms,step_ms,direction_launches,tail_launches,"
              "captures,updates,rebuilds,fallbacks,nonfinite_vertices,"
              "exit_residual,exit_relative_initial,exit_budget,exit_beta,"
              "budget_active,budget_valid,bounds_valid,residual_evaluations,target_changes,exit_reason,intersection,"
              "first_toi_ready_k,exit_k,newton_solves,tail_iterations,tail_ms,residual_rejections,history_rejections,guard_rejections,diagnostic_budget_hits,"
              "gradient_hessian_ms,residual_check_ms,linear_system_ms,ccd_ms,line_search_ms,graph_prepare_ms\n";
    frames << std::setprecision(17);
    bool valid = true;
    for(int f = 1; f <= bench::options.frames; ++f)
    {
        bench::frame = {};
        bench::frame_id = f;
        ipc.IPC_Solver(d_tetMesh);
        // Existing safety query, diagnostic only and outside all physics timers.
        const bool intersection = ipc.isIntersected(d_tetMesh);
        CUDA_SAFE_CALL(cudaMemcpy(vertices.data(), ipc._vertexes,
                                  vertices.size() * sizeof(double3), cudaMemcpyDeviceToHost));
        size_t nonfinite = 0;
        for(const auto& vertex : vertices)
            nonfinite += !(std::isfinite(vertex.x) && std::isfinite(vertex.y)
                            && std::isfinite(vertex.z));
        valid &= nonfinite == 0 && bench::frame.newton_cap_hits == 0
                 && bench::frame.diagnostic_budget_hits == 0;
        if(states.is_open())
        {
            states.write(reinterpret_cast<const char*>(vertices.data()),
                         vertices.size() * sizeof(double3));
            CUDA_SAFE_CALL(cudaMemcpy(velocities.data(), d_tetMesh.velocities,
                velocities.size() * sizeof(double3), cudaMemcpyDeviceToHost));
            velocity_states.write(reinterpret_cast<const char*>(velocities.data()),
                velocities.size() * sizeof(double3));
        }
        const auto& row = bench::frame;
        frames << f << ',' << row.newton_iterations << ',' << row.newton_cap_hits
               << ',' << row.pcg_iterations.size() << ','
               << std::accumulate(row.pcg_iterations.begin(), row.pcg_iterations.end(), uint64_t{0})
               << ',' << row.newton_ms << ',' << row.pcg_ms << ',' << row.step_ms
               << ',' << row.direction_launches << ',' << row.tail_launches
               << ',' << row.captures << ',' << row.updates << ',' << row.rebuilds
               << ',' << row.fallbacks << ',' << nonfinite
               << ',' << row.exit_residual << ',' << row.exit_relative_initial
               << ',' << row.exit_budget << ',' << row.exit_beta
               << ',' << row.budget_active << ',' << row.budget_valid
               << ',' << row.bounds_valid << ',' << row.residual_evaluations
               << ',' << row.target_changes << ',' << row.exit_reason << ',' << intersection
               << ',' << row.first_toi_ready_k << ',' << row.exit_k << ',' << row.newton_solves
               << ',' << row.tail_iterations << ',' << row.tail_ms << ',' << row.residual_rejections
               << ',' << row.history_rejections << ',' << row.guard_rejections
               << ',' << row.diagnostic_budget_hits
               << ',' << row.gradient_hessian_ms << ',' << row.residual_check_ms
               << ',' << row.linear_system_ms << ',' << row.ccd_ms << ',' << row.line_search_ms
               << ',' << row.graph_prepare_ms << '\n';
        traces << gipc::Json{{"frame", f}, {"iterations", row.pcg_iterations}}.dump() << '\n';
        if(termination_trace.is_open())
            termination_trace << gipc::Json{{"frame", f}, {"exit_reason", row.exit_reason},
                {"samples", row.termination_samples}}.dump() << '\n';
        if(newton_trace.is_open())
            newton_trace << gipc::Json{{"frame", f},
                {"physical_frame", ipc.physical_frame()},
                {"exit_reason", row.exit_reason},
                {"samples", row.newton_samples}}.dump() << '\n';
        if(stage_profile.is_open())
        {
            const auto gradient_subtotal = row.gradient_zero_ms
                + row.gradient_kinetic_ms + row.gradient_barrier_ms
                + row.gradient_friction_grad_ms + row.gradient_friction_hess_ms
                + row.gradient_ground_ms + row.gradient_contact_classify_ms
                + row.gradient_abd_ms + row.gradient_contact_index_ms
                + row.gradient_tet_fem_ms + row.gradient_bending_ms
                + row.gradient_triangle_fem_ms
                + row.gradient_soft_boundary_mass_ms;
            const auto linear_subtotal = row.linear_info_ms + row.linear_resize_ms
                + row.linear_subsystem_assemble_ms + row.linear_preconditioner_pre_ms
                + row.linear_convert_ms + row.linear_preconditioner_global_ms
                + row.linear_preconditioner_post_ms + row.linear_solver_only_ms
                + row.linear_distribute_ms;
            const auto convert_subtotal = row.convert_hash_index_ms
                + row.convert_radix_sort_ms + row.convert_gather_ms
                + row.convert_partition_mark_ms + row.convert_partition_scan_ms
                + row.convert_row_col_ms + row.convert_host_read_ms
                + row.convert_block_zero_ms + row.convert_reduce_ms
                + row.convert_output_copy_ms;
            stage_profile << gipc::Json{
                {"frame", f},
                {"gradient_hessian", {
                    {"inclusive_ms", row.gradient_hessian_ms},
                    {"subprofile_ms", gradient_subtotal},
                    {"unattributed_ms", row.gradient_hessian_ms - gradient_subtotal},
                    {"zero", row.gradient_zero_ms},
                    {"kinetic", row.gradient_kinetic_ms},
                    {"barrier", row.gradient_barrier_ms},
                    {"friction_gradient", row.gradient_friction_grad_ms},
                    {"friction_hessian", row.gradient_friction_hess_ms},
                    {"ground", row.gradient_ground_ms},
                    {"contact_classify", row.gradient_contact_classify_ms},
                    {"abd", row.gradient_abd_ms},
                    {"contact_index", row.gradient_contact_index_ms},
                    {"tet_fem_fused", row.gradient_tet_fem_ms},
                    {"cloth_bending_fused", row.gradient_bending_ms},
                    {"triangle_fem_fused", row.gradient_triangle_fem_ms},
                    {"soft_boundary_mass", row.gradient_soft_boundary_mass_ms}}},
                {"linear_system", {
                    {"inclusive_ms", row.linear_system_ms},
                    {"subprofile_ms", linear_subtotal},
                    {"unattributed_ms", row.linear_system_ms - linear_subtotal},
                    {"info_offsets", row.linear_info_ms},
                    {"resize", row.linear_resize_ms},
                    {"subsystem_assemble", row.linear_subsystem_assemble_ms},
                    {"preconditioner_pre_convert", row.linear_preconditioner_pre_ms},
                    {"convert", row.linear_convert_ms},
                    {"preconditioner_global", row.linear_preconditioner_global_ms},
                    {"preconditioner_post_convert", row.linear_preconditioner_post_ms},
                    {"pcg_solver", row.linear_solver_only_ms},
                    {"distribute_wait", row.linear_distribute_ms}}},
                {"converter", {
                    {"inclusive_ms", row.linear_convert_ms},
                    {"subprofile_ms", convert_subtotal},
                    {"unattributed_ms", row.linear_convert_ms - convert_subtotal},
                    {"hash_index", row.convert_hash_index_ms},
                    {"radix_sort", row.convert_radix_sort_ms},
                    {"value_gather", row.convert_gather_ms},
                    {"partition_mark", row.convert_partition_mark_ms},
                    {"partition_scan", row.convert_partition_scan_ms},
                    {"row_col_scatter", row.convert_row_col_ms},
                    {"unique_count_host_read", row.convert_host_read_ms},
                    {"block_zero", row.convert_block_zero_ms},
                    {"segmental_reduce", row.convert_reduce_ms},
                    {"unique_output_copy", row.convert_output_copy_ms}}},
                {"counts", {
                    {"gradient_hessian_calls", row.gradient_hessian_calls},
                    {"linear_build_calls", row.linear_build_calls},
                    {"convert_calls", row.convert_calls},
                    {"profiled_sync_calls", row.profiled_sync_calls},
                    {"profiled_d2h_calls", row.profiled_d2h_calls},
                    {"profiled_d2h_bytes", row.profiled_d2h_bytes},
                    {"profiled_resize_calls", row.profiled_resize_calls},
                    {"profiled_resize_changes", row.profiled_resize_changes},
                    {"profiled_allocation_calls", row.profiled_allocation_calls},
                    {"profiled_allocation_bytes", row.profiled_allocation_bytes},
                    {"contact_triplets", row.profiled_contact_triplets},
                    {"total_triplets", row.profiled_total_triplets},
                    {"convert_input_triplets", row.profiled_convert_input_triplets},
                    {"rhs_dofs", row.profiled_rhs_dofs},
                    {"unique_blocks", row.profiled_unique_blocks},
                    {"fem_tets", row.profiled_fem_tets},
                    {"tri_edges", row.profiled_tri_edges},
                    {"triangles", row.profiled_triangles},
                    {"fem_points", row.profiled_fem_points}}}}
                .dump() << '\n';
        }
        frames.flush();
        if(!bench::options.checkpoint_out.empty()
           && bench::options.checkpoint_frame == f)
            ipc.save_checkpoint(d_tetMesh, bench::options.checkpoint_out);
        if(!bench::options.checkpoint_output_root.empty()
           && std::find(bench::options.checkpoint_frames.begin(),
                        bench::options.checkpoint_frames.end(),
                        f)
                  != bench::options.checkpoint_frames.end())
        {
            const auto checkpoint =
                std::filesystem::path(bench::options.checkpoint_output_root)
                / ("S" + std::to_string(ipc.physical_frame()));
            ipc.save_checkpoint(d_tetMesh, checkpoint.string());
        }
        if(f % 10 == 0)
            std::cout << "Benchmark scene " << bench::options.scene << " frame "
                      << f << '/' << bench::options.frames << std::endl;
        if(!valid)
            break;
    }
    write_vertices(output / "final.f64");
    CUDA_SAFE_CALL(cudaMemcpy(velocities.data(), d_tetMesh.velocities,
                              velocities.size() * sizeof(double3), cudaMemcpyDeviceToHost));
    for(const auto& v : velocities)
        valid &= std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
    std::ofstream velocity_file(output / "final_velocities.f64", std::ios::binary);
    velocity_file.write(reinterpret_cast<const char*>(velocities.data()),
                        velocities.size() * sizeof(double3));
    std::ofstream(output / "audits.json") << bench::audits.dump(2);
    if(!bench::options.checkpoint_out.empty()
       && bench::options.checkpoint_frame < 0)
        ipc.save_checkpoint(d_tetMesh, bench::options.checkpoint_out);
    std::ofstream(output / "status.json") << gipc::Json{
        {"completed", valid && bench::frame_id == bench::options.frames},
        {"frames_completed", bench::frame_id}, {"finite_state", valid}}.dump(2);
    return valid ? 0 : 2;
}

int main(int argc, char** argv)
{
    try
    {
        for(int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];
            auto value = [&]() -> std::string
            {
                if(++i >= argc)
                    throw std::runtime_error("Missing argument value");
                return argv[i];
            };
            if(arg == "--scene")
                gipc::benchmark::options.scene = std::stoi(value());
            else if(arg == "--benchmark-frames")
                gipc::benchmark::options.frames = std::stoi(value());
            else if(arg == "--benchmark-output-dir")
                gipc::benchmark::options.output_dir = value();
            else if(arg == "--pcg-dual-graph")
                gipc::benchmark::options.dual_graph = true;
            else if(arg == "--export-states")
                gipc::benchmark::options.export_states = true;
            else if(arg == "--audit-pcg-frame")
                gipc::benchmark::options.audit_frame = std::stoi(value());
            else if(arg == "--termination")
                gipc::benchmark::options.termination = value();
            else if(arg == "--toi-min-updates")
                gipc::benchmark::options.toi_min_updates = std::stoi(value());
            else if(arg == "--toi-threshold")
                gipc::benchmark::options.toi_threshold = std::stod(value());
            else if(arg == "--residual-floor")
                gipc::benchmark::options.residual_floor = std::stod(value());
            else if(arg == "--trace-termination")
                gipc::benchmark::options.trace_termination = true;
            else if(arg == "--trace-newton")
                gipc::benchmark::options.trace_newton = true;
            else if(arg == "--diagnostic-max-ms")
                gipc::benchmark::options.diagnostic_max_ms = std::stod(value());
            else if(arg == "--diagnostic-max-newton")
                gipc::benchmark::options.diagnostic_max_newton = std::stoi(value());
            else if(arg == "--newton-tolerance-scale")
                gipc::benchmark::options.newton_tolerance_scale = std::stod(value());
            else if(arg == "--pcg-tolerance-scale")
                gipc::benchmark::options.pcg_tolerance_scale = std::stod(value());
            else if(arg == "--residual-threshold")
                gipc::benchmark::options.residual_threshold = std::stod(value());
            else if(arg == "--profile-stages")
                gipc::benchmark::options.profile_stages = true;
            else if(arg == "--production-measurement")
                gipc::benchmark::options.production_measurement = true;
            else if(arg == "--converter-indexed-reduce")
                gipc::benchmark::options.converter_indexed_reduce = true;
            else if(arg == "--cloth-offset-y")
                gipc::benchmark::options.cloth_offset_y = std::stod(value());
            else if(arg == "--checkpoint-in")
                gipc::benchmark::options.checkpoint_in = value();
            else if(arg == "--checkpoint-out")
                gipc::benchmark::options.checkpoint_out = value();
            else if(arg == "--checkpoint-roundtrip-out")
                gipc::benchmark::options.checkpoint_roundtrip_out = value();
            else if(arg == "--checkpoint-frame")
                gipc::benchmark::options.checkpoint_frame = std::stoi(value());
            else if(arg == "--checkpoint-output-root")
                gipc::benchmark::options.checkpoint_output_root = value();
            else if(arg == "--checkpoint-frames")
            {
                const std::string frames = value();
                size_t            begin  = 0;
                while(begin < frames.size())
                {
                    const size_t end = frames.find(',', begin);
                    const auto token = frames.substr(begin, end - begin);
                    if(token.empty())
                        throw std::runtime_error("Empty checkpoint frame");
                    gipc::benchmark::options.checkpoint_frames.push_back(
                        std::stoi(token));
                    if(end == std::string::npos)
                        break;
                    begin = end + 1;
                }
            }
            else
                throw std::runtime_error("Unknown argument: " + arg);
        }
        if(gipc::benchmark::options.scene < 0 || gipc::benchmark::options.scene > 5)
            throw std::runtime_error("Scene must be in 0..5");
        if(gipc::benchmark::options.frames < 0)
            throw std::runtime_error("Frame count must be positive");
        auto& opt = gipc::benchmark::options;
        if(opt.residual_threshold == 0) opt.residual_threshold = opt.toi_threshold;
        if(opt.termination != "original" && opt.termination != "toi"
           && opt.termination != "gated" && opt.termination != "compensated")
            throw std::runtime_error("Termination must be original/toi/gated/compensated");
        if(opt.toi_min_updates < 1 || !std::isfinite(opt.toi_threshold)
           || opt.toi_threshold <= 0 || opt.toi_threshold >= 1
           || !std::isfinite(opt.residual_floor) || opt.residual_floor <= 0
           || !std::isfinite(opt.residual_threshold) || opt.residual_threshold < opt.toi_threshold
           || !std::isfinite(opt.cloth_offset_y)
           || !std::isfinite(opt.diagnostic_max_ms) || opt.diagnostic_max_ms < 0
           || opt.diagnostic_max_newton < 0
           || !std::isfinite(opt.newton_tolerance_scale) || opt.newton_tolerance_scale <= 0
           || !std::isfinite(opt.pcg_tolerance_scale) || opt.pcg_tolerance_scale <= 0)
            throw std::runtime_error("Invalid TOI/residual/tolerance configuration");
        if(opt.checkpoint_frame == 0 || opt.checkpoint_frame < -1
           || (opt.checkpoint_frame > 0 && opt.checkpoint_out.empty())
           || (opt.checkpoint_frame > opt.frames && opt.frames > 0))
            throw std::runtime_error("Invalid checkpoint frame/output configuration");
        auto checkpoint_frames = opt.checkpoint_frames;
        std::sort(checkpoint_frames.begin(), checkpoint_frames.end());
        if((checkpoint_frames.empty() != opt.checkpoint_output_root.empty())
           || (!checkpoint_frames.empty()
               && (checkpoint_frames.front() <= 0
                   || checkpoint_frames.back() > opt.frames
                   || std::adjacent_find(checkpoint_frames.begin(),
                                         checkpoint_frames.end())
                          != checkpoint_frames.end())))
            throw std::runtime_error("Invalid checkpoint frame list/output root");
        if(gipc::benchmark::active())
        {
            if(gipc::benchmark::options.output_dir.empty())
                throw std::runtime_error("Benchmark output directory required");
            return run_benchmark();
        }
    }
    catch(const std::exception& error)
    {
        std::cerr << error.what() << std::endl;
        return 1;
    }
    argc = 1; // Project-specific switches have already been consumed.
    glutInit(&argc, argv);
    //glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);

    glutSetOption(GLUT_MULTISAMPLE, 16);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_MULTISAMPLE);

    glutInitWindowSize(window_width, window_height);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("FEM");

    init();

    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);


    glEnable(GL_MULTISAMPLE);
    glHint(GL_MULTISAMPLE_FILTER_HINT_NV, GL_NICEST);


    glutDisplayFunc(display);


    //glutDisplayFunc(display_func);
    glutReshapeFunc(reshape_func);
    glutKeyboardFunc(keyboard_func);
    glutSpecialFunc(&SpecialKey);
    glutMouseFunc(mouse_func);
    glutMotionFunc(motion_func);
    glutIdleFunc(idle_func);


    glutMainLoop();
    //return 0;
}
