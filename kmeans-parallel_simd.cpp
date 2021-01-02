// Implementation of the KMeans Algorithm
// reference: http://mnemstudio.org/clustering-k-means-example-1.htm

#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <chrono>
#include <functional>
#include <numeric>

#include <tbb/tbb.h>
#include <immintrin.h>

using namespace std;

class Point
{
private:
	int id_point, id_cluster;
	vector<double> values;
	int total_values;
	string name;

public:
	Point(int id_point, vector<double>& values, string name = "")
	{
		this->id_point = id_point;
		total_values = values.size();

		for(int i = 0; i < total_values; i++)
			this->values.push_back(values[i]);

		this->name = name;
		id_cluster = -1;
	}

	int getID()
	{
		return id_point;
	}

	void setCluster(int id_cluster)
	{
		this->id_cluster = id_cluster;
	}

	int getCluster()
	{
		return id_cluster;
	}

	double getValue(int index)
	{
		return values[index];
	}

	__m256d getValueAVX(int index)
	{
		// return _mm256_set_pd(values[index], values[index + 1], values[index + 2], values[index + 3]);
		return _mm256_loadu_pd(&values[index]);
	}

	int getTotalValues()
	{
		return total_values;
	}

	void addValue(double value)
	{
		values.push_back(value);
	}

	string getName()
	{
		return name;
	}
};

class Cluster
{
private:
	int id_cluster;
	vector<double> central_values;
	vector<Point> points;

public:
	Cluster(int id_cluster, Point point)
	{
		this->id_cluster = id_cluster;

		int total_values = point.getTotalValues();

		for(int i = 0; i < total_values; i++)
			central_values.push_back(point.getValue(i));

		// points.push_back(point);
	}

	void addPoint(Point point)
	{
		points.push_back(point);
	}

	bool removePoint(int id_point)
	{
		int total_points = points.size();

		for(int i = 0; i < total_points; i++)
		{
			if(points[i].getID() == id_point)
			{
				points.erase(points.begin() + i);
				return true;
			}
		}
		return false;
	}

	double getCentralValue(int index)
	{
		return central_values[index];
	}

	__m256d getCentralValueAVX(int index)
	{
		// return _mm256_set_pd(central_values[index], central_values[index + 1], central_values[index + 2], central_values[index + 3]);
		return _mm256_loadu_pd(&central_values[index]);
	}

	void setCentralValue(int index, double value)
	{
		central_values[index] = value;
	}

	Point getPoint(int index)
	{
		return points[index];
	}

	int getTotalPoints()
	{
		return points.size();
	}

	int getID()
	{
		return id_cluster;
	}
};

class KMeans
{
private:
	int K; // number of clusters
	int total_values, total_points, max_iterations;
	vector<Cluster> clusters;

	// return ID of nearest center (uses euclidean distance)
	int getIDNearestCenter(Point point)
	{
		double sum = 0.0, min_dist;
		int id_cluster_center = 0;

		int value;
		for(value = 0; value < total_values - 3; value+=4)
		{
			__m256d block = _mm256_sub_pd(
					clusters[0].getCentralValueAVX(value),
					point.getValueAVX(value));

			block = _mm256_mul_pd(block, block);
			block = _mm256_hadd_pd(block, block);

			double *values = (double*)&block;
			sum += values[0] + values[2];
		}

		for (value; value < total_values; value++)
		{
			sum += pow(clusters[0].getCentralValue(value) -
						   point.getValue(value), 2.0);
		}

		min_dist = sum;

		for(int i = 1; i < K; i++)
		{
			double dist;
			sum = 0.0;

			for(value = 0; value < total_values - 3; value+=4)
			{
				__m256d block = _mm256_sub_pd(
					clusters[i].getCentralValueAVX(value),
					point.getValueAVX(value));
				block = _mm256_mul_pd(block, block);

				double *values = (double*)&block;
				sum += values[0] + values[1] + values[2] + values[3];
			}

			for (value; value < total_values; value++)
			{
				sum += pow(clusters[i].getCentralValue(value) -
								point.getValue(value), 2.0);
			}

			dist = sum;

			if(dist < min_dist)
			{
				min_dist = dist;
				id_cluster_center = i;
			}
		}

		return id_cluster_center;
	}

public:
	KMeans(int K, int total_points, int total_values, int max_iterations)
	{
		this->K = K;
		this->total_points = total_points;
		this->total_values = total_values;
		this->max_iterations = max_iterations;
	}

	void run(vector<Point> & points)
	{
        auto begin = chrono::high_resolution_clock::now();
        
		if(K > total_points)
			return;

		vector<int> prohibited_indexes;

		// choose K distinct values for the centers of the clusters
		for(int i = 0; i < K; i++)
		{
			while(true)
			{
				int index_point = rand() % total_points;

				if(find(prohibited_indexes.begin(), prohibited_indexes.end(),
						index_point) == prohibited_indexes.end())
				{
					prohibited_indexes.push_back(index_point);
					points[index_point].setCluster(i);
					Cluster cluster(i, points[index_point]);
					clusters.push_back(cluster);
					break;
				}
			}
		}
    		auto end_phase1 = chrono::high_resolution_clock::now();
        
		int iter = 1;

				auto total_1 = 0l;
				auto total_2 = 0l;
		while(true)
		{
			// Associates each point to the nearest center
			// Sets cluster value on points only
			auto s1 = chrono::high_resolution_clock::now();
			bool done = tbb::parallel_reduce(
				tbb::blocked_range<int>(0, total_points),
				bool(true),
				[&] (
					tbb::blocked_range<int> r,
					bool in
				) {
					for(int i = r.begin(); i != r.end(); i++)
					{
						int id_old_cluster = points[i].getCluster();
						int id_nearest_center = getIDNearestCenter(points[i]);

						if(id_old_cluster != id_nearest_center)
						{
							points[i].setCluster(id_nearest_center);
							in = false;
						}
					}

					return in;
				},
				std::logical_and<bool>()
			);
			auto s2 = chrono::high_resolution_clock::now();

			// Evaluate clusters by points
			vector<int> cluster_point_count(K);
			vector<vector<double>> cluster_sum_value;

			cluster_sum_value.resize(K, vector<double>(total_values));

			for(int i = 0; i < total_points; i++)
			{
				int cluster_id = points[i].getCluster();
				cluster_point_count[cluster_id]++;

				for(int j = 0; j < total_values; j++)
				{
					cluster_sum_value[cluster_id][j] += points[i].getValue(j);
				}
			}

			// Set values for the centers of clusters
			for(int i = 0; i < K; i++)
			{
				if (cluster_point_count[i] > 0)
				{
					for(int j = 0; j < total_values; j++)
					{
						clusters[i].setCentralValue(j, cluster_sum_value[i][j] / cluster_point_count[i]);
					}
				}
			}

			if(done == true || iter >= max_iterations)
			{
				cout << "Break in iteration " << iter << "\n\n";
				break;
			}

			iter++;
			auto s3 = chrono::high_resolution_clock::now();

			total_1 += std::chrono::duration_cast<std::chrono::microseconds>(s2-s1).count();
			total_2 += std::chrono::duration_cast<std::chrono::microseconds>(s3-s2).count();
		}

		// Add points to final clusters
		for(int i = 0; i < total_points; i++)
		{
			if (points[i].getCluster() != -1)
			{
				clusters[points[i].getCluster()].addPoint(points[i]);
			}
		}


        auto end = chrono::high_resolution_clock::now();

		// shows elements of clusters
		for(int i = 0; i < K; i++)
		{
			int total_points_cluster =  clusters[i].getTotalPoints();

			cout << "Cluster " << clusters[i].getID() + 1 << endl;
			for(int j = 0; j < total_points_cluster; j++)
			{
				cout << "Point " << clusters[i].getPoint(j).getID() + 1 << ": ";
				for(int p = 0; p < total_values; p++)
					cout << clusters[i].getPoint(j).getValue(p) << " ";

				string point_name = clusters[i].getPoint(j).getName();

				if(point_name != "")
					cout << "- " << point_name;

				cout << endl;
			}

			cout << "Cluster values: ";

			for(int j = 0; j < total_values; j++)
				cout << clusters[i].getCentralValue(j) << " ";

			cout << "\n\n";
		}
		
		cout << "TOTAL EXECUTION TIME = "<<std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()<<"\n";
		
		cout << "TIME PHASE 1 = "<<std::chrono::duration_cast<std::chrono::microseconds>(end_phase1-begin).count()<<"\n";
		
		cout << "TIME PHASE 2 = "<<std::chrono::duration_cast<std::chrono::microseconds>(end-end_phase1).count()<<"\n";

		cout << "TOTAL 1 = "<< total_1 <<"\n";
		
		cout << "TOTAL 2 = "<<  total_2 <<"\n";

		cout << "Iters = " << iter << "\n"; 
	}
};

int main(int argc, char *argv[])
{
	srand (0);

	int total_points, total_values, K, max_iterations, has_name;

	cin >> total_points >> total_values >> K >> max_iterations >> has_name;

	vector<Point> points;
	string point_name;

	for(int i = 0; i < total_points; i++)
	{
		vector<double> values;

		for(int j = 0; j < total_values; j++)
		{
			double value;
			cin >> value;
			values.push_back(value);
		}

		if(has_name)
		{
			cin >> point_name;
			Point p(i, values, point_name);
			points.push_back(p);
		}
		else
		{
			Point p(i, values);
			points.push_back(p);
		}
	}

	KMeans kmeans(K, total_points, total_values, max_iterations);
	kmeans.run(points);

	return 0;
}
