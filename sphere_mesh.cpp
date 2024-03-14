#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <stdio.h>
#include <cstdlib>
#include <cmath>
#include <utility>
#include <cstdio>
#include <chrono>

using namespace std;

typedef vector<double> vertex;
typedef vector<int> face;

int n, m;
int h_max = 100;
int h_min = 0;
double** vertices;
vector<face> faces;
vector<vector<int>> conexions;
vector<vector<int>> edges;

ofstream output;

vector<int> add( vector<int> e , int b ){
    bool in = false;
    for( int v : e ){
        if( v == b ){ in = true; }
    }
    if( not in ){ e.insert(e.end() , b ); }
    return e;
}

void read_mesh(){
    //Read the file with the data of the mesh 
    cin >> n >> m ;
    double x , y , z;
    vertices = new double *[n];
    vertex v;
    for( int i = 0 ; i < n ; i++ ){
        cin >> x >> y >> z;
        if( z < h_max ){ h_max = z; }
        if( z > h_min ){ h_min = z; }
        v = {x,y,z};
        vertices[i] = new double[3];
        for( int j = 0 ; j < 3 ; j++ )
            vertices[i][j] = v[j];
        /*
        if( i < 100 ){
            cout << setprecision(19) <<  x <<" " << setprecision(19) << y << " " << setprecision(19) << z <<endl;
        }
        */
    }
    //*vertices = *vert;

    face f;
    conexions.resize(n);
    edges.resize(n);
    for( int i = 0 ; i < m ; i++ ){
        f = { 0 , 0 , 0 };
        cin >> f[0] >> f[1] >> f[2];
        for( int j = 0 ; j < 3 ; j++ ){
            f[j] = f[j]-1;
            conexions[f[j]].insert( conexions[f[j]].end() , i );
            for( int k = j-1 ; k > -1 ; k-- ){
                edges[f[j]] = add( edges[f[j]] , f[k] );
                edges[f[k]] = add( edges[f[k]] , f[j] );
            }
        }
        faces.insert( faces.end() , f );
    }
}

vector<pair<int,double>> cylinder_mesh_intersection( vertex p , vector<double> d , int r  ){
    vector<pair<int,double>> cylinder;
    double b , t2 ;
    
    int i = -1;
    //auto t1 = std::chrono::system_clock::now();
    for( int i = 0 ; i < n ; i++ ){
        //double v[3] = { vertices[i][0] , vertices[i][1] , vertices[i][2] };
        /*
        if( i % 1000 == 1)
            t1 = std::chrono::system_clock::now();
        */
        double x[3] = { p[0]-vertices[i][0] , p[1]-vertices[i][1] , p[2]-vertices[i][2] };
        b = x[0]*d[0] + x[1]*d[1] + x[2]*d[2];
        t2 = r*r-x[0]*x[0]-x[1]*x[1]-x[2]*x[2]+b*b;
        if( t2 > 0 ){
                cylinder.insert( cylinder.end() , make_pair(i,-b-sqrt(t2)) );
        }

        /*
        if( i % 1000 == 999 ){
            auto t4 = std::chrono::system_clock::now();

            std::chrono::duration<double> proc1 = t4-t1;

            cout << proc1.count() << endl;
        }
        */
    }
    return cylinder;
}

vector<vector<double>> inverse(vector<vector<double>> mat){
    int i, j;
    vector<vector<double>> inv ={{0,0,0},{0,0,0},{0,0,0}};
    float determinant = 0;
    for(i = 0; i < 3; i++)
        determinant = determinant + (mat[0][i] * (mat[1][(i+1)%3] * mat[2][(i+2)%3] - mat[1][(i+2)%3] * mat[2][(i+1)%3]));

    for(i = 0; i < 3; i++){
        for(j = 0; j < 3; j++)
            inv[i][j] = ((mat[(j+1)%3][(i+1)%3] * mat[(j+2)%3][(i+2)%3]) - (mat[(j+1)%3][(i+2)%3] * mat[(j+2)%3][(i+1)%3]));
    }
    return inv;
}

vector<double> cross( vector<double> v1 , vector<double> v2 ){
    vector<double> normal = { v1[1]*v2[2]-v1[2]*v2[1] , v1[2]*v2[0]-v1[0]*v2[2] , v1[0]*v2[1]-v1[1]*v2[0] };
    double normal_l = sqrt( normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2] );
    for( int i = 0 ;i < 3 ; i++ )
        normal[i] = normal[i]/normal_l;
    return normal;
}

vector<double> Moller_Trumbore(  vector<vertex> triangle , vertex p , vector<double> d , int r  ){
    vector<double> v1(3);
    vector<double> v2(3);
    vector<double> n(3);
    for( int i = 0 ; i < 3 ; i++ ){
        v1[i] = triangle[1][i] - triangle[0][i];
        v2[i] = triangle[2][i] - triangle[0][i];
    }
    vector<double> normal = cross(v1,v2);
    vertex p1 = { p[0] - normal[0]*r - triangle[0][0] , p[1] - normal[1]*r - triangle[0][1] , p[2] - normal[2]*r  - triangle[0][2]};
    vector<vector<double>> M = { {-d[0] , v1[0] ,v2[0]} , {-d[1] , v1[1] , v2[1] } , {-d[2],v1[2],v2[2]} };
    M = inverse(M);
    vector<double> result = { M[0][0] * p1[0] , M[0][1] * p1[1] , M[0][2] * p1[2] , M[1][0] * p1[0] , M[1][1] * p1[1] , M[1][2] * p1[2] , M[2][0] * p1[0] , M[2][1] * p1[1] , M[2][2] * p1[2] };
    return result ; 
}

double sphere_line_intersection( vertex a , vertex b , vertex p , vector<double> d , int r ){
    vertex v = { b[0]-a[0] , b[1]-a[1] , b[2]-a[2] };
    vector<double> normal = cross( v , d );
    double e = (p[0]-a[0])*normal[0] + (p[1]-a[1])*normal[1] + (p[2]-a[2])*normal[2]; 
    if( e*e > r*r )
        return 10000;
    double r1 = sqrt( r*r - e*e);
    vertex p1 = { p[0]-e*normal[0] , p[1]-e*normal[1] , p[2]-e*normal[2] };
    vector<double> s = cross( v , normal );
    if( s[0]*s[0] + s[1]*s[1] + s[2]*s[2] == 0 )
        return 10000;
    if( s[0]*d[0] + s[1]*d[1] + s[2]*d[2] < 0 )
        s = { -s[0] , -s[1] , -s[2] };
    vertex p2 = { p1[0]+r1*s[0] , p1[1]+r1*s[1] , p1[2]+r1*s[2] };
    
    if ((v[0]*d[1]-v[1]*d[0])==0) 
        return 1000;
    double t_int = (v[1]*(p2[0]-a[0]) - v[0]*(p2[1]-a[1]))/(v[0]*d[1]-v[1]*d[0]);
    double g;
    if (v[0]!= 0)
        g = (p2[0]-a[0]+t_int*d[0])/v[0];
    else if (v[1] != 0)
        g = (p2[1] - a[1] + t_int * d[1]) / v[1];
    else if (v[2] != 0)
        g = (p2[2] - a[2] + t_int * d[2]) / v[2];

    if( g < 0 or g > 1 )
        return 10000;
    return t_int ; 
}

void particle_mesh_intersection( vertex p , vector<double> d , int r ){
    //auto t1 = std::chrono::system_clock::now();
    vector<pair<int,double>> cylinder = cylinder_mesh_intersection( p , d , r  );
    //auto t4 = std::chrono::system_clock::now();
    //std::chrono::duration<double> proc1 = t4-t1;
    //cout << proc1.count() << endl;
    
    int kind = 0;
    double t_min = 10000;
    int p_min = -1;
    int p_aux = -1;
    for( pair<int,double> c : cylinder ){
        if( c.second < t_min ){ 
            p_min = c.first;
            t_min = c.second; 
        }
    }
    for( pair<int,double> c : cylinder ){
        if( c.second < t_min + 1 ){
            for( int f : conexions[c.first] ){
                vector<double> MT = Moller_Trumbore( { {vertices[faces[f][0]][0] , vertices[faces[f][0]][1] , vertices[faces[f][0]][2] }
                                                      ,{vertices[faces[f][1]][0] , vertices[faces[f][1]][1] , vertices[faces[f][1]][2] }
                                                      ,{vertices[faces[f][2]][0] , vertices[faces[f][2]][1] , vertices[faces[f][2]][2] }} , p , d , r );
                if( MT[0] < t_min and MT[1] > 0 and MT[2] > 0 and MT[1]+MT[2] < 1 ){
                    kind = 1 ;
                    t_min = MT[0];
                    p_min = f;
                }
            }
            for( int e : edges[c.first] ){
                vertex a = { vertices[c.first][0] , vertices[c.first][1] , vertices[c.first][2] };
                vertex b = { vertices[e][0] , vertices[e][1] , vertices[e][2] };
                double t = sphere_line_intersection( a , b , p , d , r );
                if (t < t_min){
                    kind = 2;
                    t_min = t;
                    p_min = c.first;
                    p_aux = e;
                }
            }
        }
    }

    //auto t5 = std::chrono::system_clock::now();
    //std::chrono::duration<double> proc2 = t5-t4;
    //cout << proc1.count() << " " << proc2.count() << endl;

    double margin = sqrt( r*r - (p[2]+t_min*d[2]-h_max)*(p[2]+t_min*d[2]-h_max) );


    if( not (p[0] + t_min * d[0] < margin or p[0] + t_min * d[0] > 1411 - margin or p[1] + t_min * d[1] < margin or p[1] + t_min * d[1] > 1057 - margin)){
        #pragma omp critical
        {
        //cout << kind << " " << p_min << " " << t_min << " " ;//<< proc1.count() << " ";
        char o [50];
        sprintf( o , "%d %d %d \n" , kind , p_min , p_aux  ) ;
        string o2 = o;
        output.write( o2.c_str() , o2.size() );
        //if( kind == 2 )
        //    cout << p_aux ;
        //cout << endl;
        }    
    }
}

void mesh_characterization( int r , int  phi , int theta , int a , int b ){ 
    vertex p;
    if( b > m ){ b = m; }
    vector<double> d = { sin(phi/180.*M_PI)*cos(theta/180.*M_PI), 
                         sin(phi/180.*M_PI)*sin(theta/180.*M_PI),
                         -cos(phi/180.*M_PI) };
    #pragma omp parallel for private( p )
    for( int f = a ; f < b ; f++ ){
        p = {0,0,0};
        for( int v : faces[f] ){
            vertex s = { vertices[v][0] , vertices[v][1] , vertices[v][2] } ;
            for( int i = 0 ; i < 3 ; i++ ){ p[i] += s[i]/3.; }
        }
        //cout << f << endl;
        particle_mesh_intersection( p , d , r );
    } 
}


int main( int argc , char *argv[] ){
    read_mesh() ;

    char filename [50];
    int f = sprintf( filename , "out/normals_%s_%s_%s.bin" , argv[1] , argv[2] , argv[3]  );
    output.open( filename , ios::out | ios::binary );

    mesh_characterization( atoi( argv[1]) , atoi( argv[2]) , atoi( argv[3]) , 0 , 10000  );
    
    output.close();
}