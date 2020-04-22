// PCL lib Functions for processing point clouds 

#include "processPointClouds.h"

//constructor:
template<typename PointT>
ProcessPointClouds<PointT>::ProcessPointClouds() {}


//de-constructor:
template<typename PointT>
ProcessPointClouds<PointT>::~ProcessPointClouds() {}


template<typename PointT>
void ProcessPointClouds<PointT>::numPoints(typename pcl::PointCloud<PointT>::Ptr cloud)
{
    std::cout << cloud->points.size() << std::endl;
}


template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::FilterCloud(typename pcl::PointCloud<PointT>::Ptr cloud, float filterRes, Eigen::Vector4f minPoint, Eigen::Vector4f maxPoint)
{
    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();

    pcl::VoxelGrid<PointT> vg;
  	typename pcl::PointCloud<PointT>::Ptr cloudFiltered(new pcl::PointCloud<PointT>);
  
  	vg.setInputCloud(cloud);
  	vg.setLeafSize(filterRes, filterRes, filterRes);
  	vg.filter(*cloudFiltered);
  
  	typename pcl::PointCloud<PointT>::Ptr cloudRegion (new pcl::PointCloud<PointT>);
  
  	pcl::CropBox<PointT> region(true);
  	region.setMin(minPoint);
  	region.setMax(maxPoint);
  	region.setInputCloud(cloudFiltered);
  	region.filter(*cloudRegion);
  
  	std::vector<int> indices;

  	pcl::CropBox<PointT> roof(true);
  	region.setMin(Eigen::Vector4f (-1.5, -1.7, -1, 1));
  	region.setMax(Eigen::Vector4f(2.6, 1.7, -4, 1));
  	region.setInputCloud(cloudRegion);
  	region.filter(indices);
  
  	pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  	for (int point : indices)
    	inliers->indices.push_back(point);
  
  	pcl::ExtractIndices<PointT> extract;
  	extract.setInputCloud(cloudRegion);
  	extract.setIndices(inliers);
  	extract.setNegative(true);
  	extract.filter(*cloudRegion);
  
    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "filtering took " << elapsedTime.count() << " milliseconds" << std::endl;

    return cloudRegion;
}


template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SeparateClouds(pcl::PointIndices::Ptr inliers, typename pcl::PointCloud<PointT>::Ptr cloud) 
{
  	typename pcl::PointCloud<PointT>::Ptr objectCloud(new pcl::PointCloud<PointT>());
	typename pcl::PointCloud<PointT>::Ptr planeCloud(new pcl::PointCloud<PointT>());
  
  	for (int idx : inliers->indices)
  	{
    	planeCloud->points.push_back(cloud->points[idx]);
  	}
  
   	pcl::ExtractIndices<PointT> extract;
	extract.setInputCloud (cloud);
    extract.setIndices (inliers);
    extract.setNegative (true);
    extract.filter (*objectCloud);
  
    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult(objectCloud,planeCloud);
    return segResult;
}


template<typename PointT>
void ProcessPointClouds<PointT>::RansacPlane(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceTol, pcl::PointIndices::Ptr inliersResult)
{
	float x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4;
      
	while (maxIterations--)
    {
		std::unordered_set<int> inliers;

		// Randomly sample subset and fit plane
      	while (inliers.size() < 3)
        	inliers.insert(rand()%(cloud->points.size()));
        	
      	auto itr = inliers.begin();

      	x1 = cloud->points[*itr].x;
      	y1 = cloud->points[*itr].y;
      	z1 = cloud->points[*itr].z;
      	itr++;
      	
		x2 = cloud->points[*itr].x;
      	y2 = cloud->points[*itr].y;
      	z2 = cloud->points[*itr].z;
      	itr++;
      	
		x3 = cloud->points[*itr].x;
      	y3 = cloud->points[*itr].y;
      	z3 = cloud->points[*itr].z;
      
      	float vx1 = x2-x1;
      	float vy1 = y2-y1;
      	float vz1 = z2-z1;
      
      	float vx2 = x3-x1;
      	float vy2 = y3-y1;
      	float vz2 = z3-z1;

      	float i = vy1*vz2-vz1*vy2;
      	float j = vz1*vx2-vx1*vz2;
      	float k = vx1*vy2-vy1*vx2;

      	float d = -(i*x1+j*y1+k*z1);
		float distDen = sqrt(i*i+j*j+k*k);
      
      	for (int index=0; index < cloud->points.size(); index++)
        {
          	if (inliers.count(index))
              continue;
          
          	PointT point = cloud->points[index];
          	x4 = point.x;
          	y4 = point.y;
          	z4 = point.z;
          
          	float dist = fabs(i*x4+j*y4+k*z4+d)/distDen;
          
          	if (dist <= distanceTol)
            	inliers.insert(index);  
        }

      	if (inliers.size() > inliersResult->indices.size())
        {
			inliersResult->indices.clear();
			for (int i : inliers)
	        	inliersResult->indices.push_back(i);
        }
    }
}


template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SegmentPlane(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceThreshold)
{
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
	RansacPlane(cloud, maxIterations, distanceThreshold, inliers);

  	if (inliers->indices.size () == 0)
    {
    	std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
  	}

    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult = SeparateClouds(inliers,cloud);
    return segResult;
}


template<typename PointT>
void ProcessPointClouds<PointT>::ClusterHelper(int indice, const std::vector<std::vector<float>>& points, std::vector<int>& cluster, std::vector<bool>& processed, KdTree* tree, float distanceTol)
{
	processed[indice] = true;

	cluster.push_back(indice);

	std::vector<int> nearest = tree->search(points[indice], distanceTol);

	for (int id : nearest)
	{
		if (!processed[id])
			ClusterHelper(id, points, cluster, processed, tree, distanceTol);
	}
}


template<typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::Clustering(typename pcl::PointCloud<PointT>::Ptr cloud, float clusterTolerance, int minSize, int maxSize)
{
    // Time clustering process
    auto startTime = std::chrono::steady_clock::now();

	KdTree* tree = new KdTree;
	std::vector<std::vector<float>> cloudPoints;

    for (int i=0; i<cloud->points.size(); i++) 
	{
		std::vector<float> p = {cloud->points[i].x, cloud->points[i].y, cloud->points[i].z};
    	tree->insert(p, i); 
		cloudPoints.push_back(p);
	}

  	std::vector<std::vector<int>> clusters;
	std::vector<bool> processed(cloud->points.size(), false);

	int i = 0;
	while (i < cloud->points.size())
	{
		if (processed[i])
		{
			i++;
			continue;
		}

		std::vector<int> clusterIds;
		ClusterHelper(i, cloudPoints, clusterIds, processed, tree, clusterTolerance);
		clusters.push_back(clusterIds);
		i++;
	}
  
  	std::vector<typename pcl::PointCloud<PointT>::Ptr> cloudClusters;

  	for (std::vector<int> cluster: clusters)
    {     
		if (cluster.size() < minSize || cluster.size() > maxSize)
			continue;

		typename pcl::PointCloud<PointT>::Ptr cloudCluster(new pcl::PointCloud<PointT>);

		for (int index : cluster)
			cloudCluster->points.push_back(cloud->points[index]);

      	cloudCluster->width = cloudCluster->points.size();
      	cloudCluster->height = 1;
      	cloudCluster->is_dense = true;

		cloudClusters.push_back(cloudCluster);
    }
  
    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "clustering took " << elapsedTime.count() << " milliseconds and found " << cloudClusters.size() << " clusters" << std::endl;

    return cloudClusters;
}


template<typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::ClusteringPcl(typename pcl::PointCloud<PointT>::Ptr cloud, float clusterTolerance, int minSize, int maxSize)
{
    // Time clustering process
    auto startTime = std::chrono::steady_clock::now();

    std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;

  	typename pcl::search::KdTree<PointT>::Ptr tree (new typename pcl::search::KdTree<PointT>);
  	tree->setInputCloud(cloud);

  	std::vector<pcl::PointIndices> clusterIndices;
  	pcl::EuclideanClusterExtraction<PointT> ec;

  	ec.setClusterTolerance(clusterTolerance); 
  	ec.setMinClusterSize(minSize);
  	ec.setMaxClusterSize(maxSize);
  	ec.setSearchMethod(tree);
  	ec.setInputCloud(cloud);
  	ec.extract(clusterIndices);
  
  	for (pcl::PointIndices getIndices: clusterIndices)
    {
      	typename pcl::PointCloud<PointT>::Ptr cloudCluster(new pcl::PointCloud<PointT>);
      
      	for (int index : getIndices.indices)
          	cloudCluster->points.push_back(cloud->points[index]);
      
      	cloudCluster->width = cloudCluster->points.size();
      	cloudCluster->height = 1;
      	cloudCluster->is_dense = true;
      
      	clusters.push_back(cloudCluster);
    }
  
    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "clustering took " << elapsedTime.count() << " milliseconds and found " << clusters.size() << " clusters" << std::endl;

    return clusters;
}


template<typename PointT>
Box ProcessPointClouds<PointT>::BoundingBox(typename pcl::PointCloud<PointT>::Ptr cluster)
{
    // Find bounding box for one of the clusters
    PointT minPoint, maxPoint;
    pcl::getMinMax3D(*cluster, minPoint, maxPoint);

    Box box;
    box.x_min = minPoint.x;
    box.y_min = minPoint.y;
    box.z_min = minPoint.z;
    box.x_max = maxPoint.x;
    box.y_max = maxPoint.y;
    box.z_max = maxPoint.z;

    return box;
}


template<typename PointT>
void ProcessPointClouds<PointT>::savePcd(typename pcl::PointCloud<PointT>::Ptr cloud, std::string file)
{
    pcl::io::savePCDFileASCII (file, *cloud);
    std::cerr << "Saved " << cloud->points.size () << " data points to "+file << std::endl;
}


template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::loadPcd(std::string file)
{
    typename pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);

    if (pcl::io::loadPCDFile<PointT> (file, *cloud) == -1) 
    {
        PCL_ERROR ("Couldn't read file \n");
    }

    std::cerr << "Loaded " << cloud->points.size () << " data points from "+file << std::endl;

    return cloud;
}


template<typename PointT>
std::vector<boost::filesystem::path> ProcessPointClouds<PointT>::streamPcd(std::string dataPath)
{
    std::vector<boost::filesystem::path> paths(boost::filesystem::directory_iterator{dataPath}, boost::filesystem::directory_iterator{});

    // sort files in accending order so playback is chronological
    sort(paths.begin(), paths.end());

    return paths;
}