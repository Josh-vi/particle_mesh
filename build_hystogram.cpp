#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <iomanip>

using namespace std;

int main( int argc , char *argv[] ){
    int r = atoi(argv[1]);
    int phi = atoi(argv[2]);
    int theta_lst[] = { 0 , 45 , 90 , 135 , 180 , 225 , 270 , 315 };
    int n = 5;
    double radious[n] = { 2 , 6 , 12 , 20 , 30 };
    int partitions[n] = { 1 , 25 , 25 , 25 , 25 };
    int cum_partitions[n] ; 
    int cum = 0;
    for( int i = 0 ; i < n ; i++ ){
        cum_partitions[i] = cum;
        cum += partitions[i];
    }
    
    int blocks = 0;
    for( double p : partitions ){ blocks += p; }
    double phi_points[blocks] , theta_points[blocks];
    phi_points[0] = 0;
    theta_points[0] = 0;
    int block_id = 0;
    for( int layer = 1 ; layer < n ; layer++ ){
	for( int block = 0 ; block < partitions[layer] ; block++ ){
	    block_id += 1;
	    phi_points[block_id] = (radious[layer]+radious[layer-1])/2.;
	    theta_points[block_id] = 360./partitions[layer]*block;
	    if( theta_points[block_id] > 180 ){ theta_points[block_id] -= 360 ; }    
	}
    }
    double prob[blocks];
    for( int i = 0 ; i < blocks ; i++ ){ prob[i] = 0;}


    char file_name[50];
    double phi_n , theta_n , block_size;
    int block_n , layer_n  ;
    int count = 0 ;
    for( int theta : theta_lst ){
	int size_s = sprintf( file_name , "output_part/normals/spheric_%d_%d_%d.txt" , r , phi , theta );
	ifstream file ; file.open( file_name );
	while( file >> phi_n >> theta_n ){
	    if( phi_n < radious[n-1] and phi_n > 0 ){
		layer_n = 0;
		for( int r : radious ){ if( phi_n > r ) layer_n += 1 ; }
		block_size = 360./partitions[layer_n];
		theta_n += block_size/2.;
		if( theta_n < 0 ){ theta_n += 360 ; }
		block_n = floor( theta_n / block_size ) + cum_partitions[layer_n];
		if( block_n >= 0 ){ 
		    prob[block_n ] += 1 ; 
		    count += 1;
		}
	    }
	}
    }
    for( int b = 0 ; b < blocks ; b ++ ){
	prob[b] = prob[b]/count;
    }

    for( double r : radious ){ cout << r << " " ; }
    cout << endl;
    for( int p : partitions ) {cout << p << " " ; }
    cout << endl;
    double cum_prob = 0;
    for( int i = 0 ; i < blocks ; i++ ){
	cum_prob += prob[i];
	cout << phi_points[i] << " " << setprecision(4)<< theta_points[i] << " " << setprecision(19) << cum_prob << endl;;
    }
}
