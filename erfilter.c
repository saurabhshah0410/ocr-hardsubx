#include <stdio.h>
#include "math.h"

// default 1st and 2nd stage classifier
typedef struct ERClassifierNM
{
    Boost* boost;
} ERClassifierNM ;

// the struct implementing the interface for the 1st and 2nd stages of Neumann and Matas algorithm
typedef struct ERFilterNM
{
    int type; // type = 1 --> evalNM1, type = 2 --> evalNM2
    float minProbability;   
    bool  nonMaxSuppression;
    float minProbabilityDiff;

    int thresholdDelta;
    float maxArea; 
    float minArea;

    ERClassifierNM* classifier;

    // count of the rejected/accepted regions
    int num_rejected_regions;
    int num_accepted_regions;

    //! Pointer to the Input/output regions
    vector* regions; //ERStat
    //! image mask used for feature calculations
    Mat region_mask;
} ERFilterNM ;

/** @brief The ERStat structure represents a class-specific Extremal Region (ER).

An ER is a 4-connected set of pixels with all its grey-level values smaller than the values in its
outer boundary. A class-specific ER is selected (using a classifier) from all the ER's in the
component tree of the image. :
 */
typedef struct ERStat
{
    //! Incrementally Computable Descriptors
    int pixel;
    int level;
    int area;
    int perimeter;
    int euler;                 //!< Euler's number
    Rect rect;                 // Rect_<int>
    double raw_moments[2];     //!< order 1 raw moments to derive the centroid
    double central_moments[3]; //!< order 2 central moments to construct the covariance matrix
    vector* crossings;         //!< horizontal crossings
    float med_crossings;       //!< median of the crossings at three different height levels

    //! stage 2 features
    float hole_area_ratio;
    float convex_hull_ratio;
    float num_inflexion_points;

    //! pixel list after the 2nd stage
    int** pixels;

    //! probability that the ER belongs to the class we are looking for
    double probability;

    //! pointers preserving the tree structure of the component tree --> might be optional
    ERStat* parent;
    ERStat* child;
    ERStat* next;
    ERStat* prev;

    //! whenever the regions is a local maxima of the probability
    bool local_maxima;
    ERStat* max_probability_ancestor;
    ERStat* min_probability_ancestor;
} ERStat ;

void init_ERStat(ERStat* erstat, int init_level, int init_pixel, int init_x, int init_y)
{
    erstat->level = init_level;
    erstat->pixel = init_pixel;
    erstat->area = 0;
    erstat->perimeter = 0;
    erstat->euler = 0;
    erstat->probability = 1.0;
    erstat->local_maxima = 0;
    erstat->parent = 0;
    erstat->child = 0;
    erstat->prev = 0;
    erstat->next = 0;
    erstat->max_probability_ancestor = 0;
    erstat->min_probability_ancestor = 0;
    erstat->rect = init_Rect(init_x, init_y, 1, 1);
    (erstat->raw_moments)[0] = 0.0;
    (erstat->raw_moments)[1] = 0.0;
    (erstat->central_moments)[0] = 0.0;
    (erstat->central_moments)[1] = 0.0;
    (erstat->central_moments)[2] = 0.0;
    erstat->crossings = malloc(sizeof(vector));
    vector_init(erstat->crossings);
    int val = 0;
    vector_add(erstat->crossings, &val);
}

// set/get methods to set the algorithm properties,
void setCallback(ERFilterNM* filter,ERClassifierNM *erc)
{
    filter->classifier = *erc;
}

void setMinArea(ERFilterNM* filter, float _minArea)
{
    filter->minArea = _minArea;
}

void setMaxArea(ERFilterNM* filter, float _maxArea)
{
    filter->maxArea = _maxArea;
}

void setMinProbability(ERFilterNM* filter, float _minProbability)
{
    filter->minProbability = _minProbability;
}

void setNonMaxSuppression(ERFilterNM* filter, bool _nonMaxSuppression)
{
    filter->nonMaxSuppression = _nonMaxSuppression;
}

void setMinProbabilityDiff(ERFilterNM* filter, float _minProbabilityDiff)
{
    filter->minProbabilityDiff = _minProbabilityDiff;
}

/*!
    Compute the different channels to be processed independently in the N&M algorithm
    Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012
*/
void computeNMChannels(Mat src, vector* _channels/* Mat */)
{
    if(empty(src)) {
        for(int i = 0; i < vector_size(_channels); i++)
        {
            Mat* m = (Mat*)vector_get(_channels, i);
            m->data = m->rows = m->cols = 0;
            m->step = 0;
        }
        vector_free(_channels);
        return;
    }

    assert(type(src) == CV_8UC3); //src.flags = 1124024336 apparently  
    createVectorMat(_channels, 5, 1, depth(src), -1); //depth = CV_8U

    vector* channelsRGB = malloc(sizeof(vector)); //Mat
    vector_init(channelsRGB);
    split(src, channelsRGB);
    for (int i = 0; i < channels(src)/*3 */; i++)
    {
        createVectorMat(_channels, src.rows, src.cols, CV_8UC1, i);
        Mat channel = *(Mat*)vector_get(_channels, i);
        copyTo(*(Mat*)vector_get(channelsRGB, i), &channel);
    }

    Mat hls;
    cvtColor(src, &hls, COLOR_RGB2HLS/*53*/);
    vector* channelsHLS = malloc(sizeof(vector));
    vector_init(channelsHLS);
    split(hls, channelsHLS);

    createVectorMat(_channels, src.rows, src.cols, CV_8UC1, 3);
    Mat* channelL = (Mat*)vector_get(_channels, 3);
    copyTo(*(Mat*)vector_get(channelsHLS, 1), channelL);

    Mat grey;
    cvtColor(src, &grey, COLOR_RGB2GRAY/*7*/);
    Mat gradient_magnitude;
    create(&gradient_magnitude, grey.rows, grey.cols, 5/*CV_32FC1*/);
    get_gradient_magnitude(grey, gradient_magnitude);
    convertTo(gradient_magnitude, &gradient_magnitude, CV_8UC1, 1, 0, CVT32F8U);

    createVectorMat(_channels, src.rows, src.cols, CV_8UC1, 4);
    Mat* channelGrad = (Mat*)vector_get(_channels, 4);
    copyTo(gradient_magnitude, channelGrad);
}

double eval_dummy(ERClassifierNM* erc, const ERStat stat)
{
    if(stat.area == 0)
        return (double)0.0;

    return (double)1.0;
}

// The classifier must return probability measure for the region --> Stage-1
float evalNM1(ERClassifierNM* erc, const ERStat stat)
{
	float sample_[4] = {(float)(stat.rect.width)/(stat.rect.height), // aspect ratio
                     sqrt((float)(stat.area))/stat.perimeter, // compactness
                     (float)(1-stat.euler), //number of holes
                     stat.med_crossings};
	Mat sample; /* flags =  */
	create(&sample, 1, 4, 5/*CV_32F1*/);
    leftshift_op(&sample, 4, sample_);
	
	float votes = predict_ml(&(erc->boost->impl), sample, 257);

	// Logistic Correction returns a probability value (in the range(0,1))
	return (double)1-(double)1/(1+exp(-2*votes));
}

// The classifier must return probability measure for the region --> Stage-2
double evalNM2(ERClassifierNM* erc, const ERStat stat)
{
	float sample_[7] = {(float)(stat.rect.width)/(stat.rect.height), // aspect ratio
                     sqrt((float)(stat.area))/stat.perimeter, // compactness
                     (float)(1-stat.euler), //number of holes
                     stat.med_crossings, stat.hole_area_ratio,
                     stat.convex_hull_ratio, stat.num_inflexion_points};
    Mat sample;
    create(&sample, 1, 7, sample_);
    leftshift_op(&sample, 7, sample_);

	float votes = predict_ml(erc->boost, sample, 257);

	// Logistic Correction returns a probability value (in the range(0,1))
	return (double)1-(double)1/(1+exp(-2*votes));
}


/*!
    Allow to implicitly load the default classifier when creating an ERFilter object.
    The function takes as parameter the XML or YAML file with the classifier model
    (e.g. trained_classifierNM1.xml) returns a pointer to ERFilter::Callback.
*/
ERClassifierNM* loadclassifierNM(ERClassifierNM* erc, const char* filename)
{
	FILE* f = fopen(filename, "r");
	if(f != NULL)
	{
		erc->boost = load_ml(filename);
		return erc;
	}
	else
		fatal("Default classifier file not found!");
}


/*!
    Create an Extremal Region Filter for the 1st stage classifier of N&M algorithm
    Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012

    The component tree of the image is extracted by a threshold increased step by step
    from 0 to 255, incrementally computable descriptors (aspect_ratio, compactness,
    number of holes, and number of horizontal crossings) are computed for each ER
    and used as features for a classifier which estimates the class-conditional
    probability P(er|character). The value of P(er|character) is tracked using the inclusion
    relation of ER across all thresholds and only the ERs which correspond to local maximum
    of the probability P(er|character) are selected (if the local maximum of the
    probability is above a global limit pmin and the difference between local maximum and
    local minimum is greater than minProbabilityDiff).

    \param  cb                Callback with the classifier.
                              default classifier can be implicitly load with function loadClassifierNM()
                              from file in samples/cpp/trained_classifierNM1.xml
    \param  thresholdDelta    Threshold step in subsequent thresholds when extracting the component tree
    \param  minArea           The minimum area (% of image size) allowed for retrieved ER's
    \param  minArea           The maximum area (% of image size) allowed for retrieved ER's
    \param  minProbability    The minimum probability P(er|character) allowed for retrieved ER's
    \param  nonMaxSuppression Whenever non-maximum suppression is done over the branch probabilities
    \param  minProbability    The minimum probability difference between local maxima and local minima ERs
*/
ERFilterNM* createERFilterNM1(ERClassifierNM* erc, int thresholdDelta, float minArea, float maxArea, float minArea, float minProbability, bool nonMaxSuppression, float minProbabilityDiff, bool is_dummy)
{
	assert((minProbability >= 0.) && (minProbability <= 1.));
	assert(minArea < maxArea) && (minArea >=0.) && (maxArea <= 1.);
	assert((thresholdDelta >= 0) && (thresholdDelta <= 128));
	assert((minProbabilityDiff >= 0.) && (minProbabilityDiff <= 1.));

	ERFilterNM* filter;
    if(is_dummy)
        filter->stage = 0;
    filter->stage = 1; //Stage 1 Extremal Region Filter
    setThresholdDeta(filter, thresholdDelta);
	setCallback(filter, erc);
	setMinArea(filter, minArea);
	setMaxArea(filter, maxArea);
	setMinProbability(filter, minProbability);
	setNonMaxSuppression(filter, nonMaxSuppression);
	setMinProbabilityDiff(filter, minProbabilityDiff);
	return filter;
}


/*!
    Create an Extremal Region Filter for the 2nd stage classifier of N&M algorithm
    Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012

    In the second stage, the ERs that passed the first stage are classified into character
    and non-character classes using more informative but also more computationally expensive
    features. The classifier uses all the features calculated in the first stage and the following
    additional features: hole area ratio, convex hull ratio, and number of outer inflexion points.

    \param  cb             Callback with the classifier
                           default classifier can be implicitly load with function loadClassifierNM()
                           from file in samples/cpp/trained_classifierNM2.xml
    \param  minProbability The minimum probability P(er|character) allowed for retreived ER's
*/
ERFilterNM* createERFilterNM2(ERClassifierNM* erc, float minProbability)
{
	assert(minProbability >= 0. && minProbability <= 1.)

	ERFilterNM* filter;
    filter->stage = 2; //Stage 2 Extremal Region Filter
	setCallback(filter, erc);
	setMinProbability(filter, minProbability);
	return filter;
}


// the key method. Takes image on input, vector of ERStat is output for the first stage,
// input/output for the second one.
void run(ERFilterNM* filter, Mat m, vector* _regions)
{
	filter->regions = _regions;
    zeros(&(filter->region_mask), m.rows+2, m.cols+2, CV_8UC1);
	
	// if regions vector is empty we must extract the entire component tree
	if(!vector_size(filter->regions))
	{
		er_tree_extract(filter, m);
		if(filter->nonMaxSuppression)
        {
            vector* aux_regions = malloc(sizeof(vector));
            vector_init(aux_regions);
            vector_swap(filter->regions, aux_regions);
            vector_init_n(filter->regions, vector_size(aux_regions));
            er_tree_nonmax_suppression(filter, (ERStat*)vector_front(aux_regions), NULL, NULL);
            vector_free(aux_regions);
        }   
	}
	else // if regions vector is already filled we'll just filter the current regions
	{
        // the tree root must have no parent
		assert((vector_front(filter->regions))->parent == NULL);

        vector* aux_regions = malloc(sizeof(vector));
        vector_init(aux_regions);
        vector_swap(filter->regions, aux_regions);
        vector_init_n(filter->regions, vector_size(aux_regions));
		er_tree_filter(filter, m, vector_front(filter->regions), NULL, NULL);
        vector_free(aux_regions);
	}
}

// extract the component tree and store all the ER regions
// uses the algorithm described in
// Linear time maximally stable extremal regions, D Nistér, H Stewénius – ECCV 2008
void er_tree_extract(ERFilterNM* filter, Mat src)
{	
	if(filter->thresholdDelta > 1)
	{
        divide_op(&src, filter->thresholdDelta);
		minus_op(&src, filter->thresholdDelta);
	}
	const unsigned char* image_data = src.data;
    int width = src.cols, height = src.rows;

    // the component stack
    vector* er_stack = malloc(sizeof(vector)); //ERStat*
    vector_init(er_stack);

    // the quads for Euler's number calculation
    // quads[2][2] and quads[2][3] are never used.
    // The four lowest bits in each quads[i][j] correspond to the 2x2 binary patterns
    // Q_1, Q_2, Q_3 in the Neumann and Matas CVPR 2012 paper
    // (see in page 4 at the end of first column).
    // Q_1 and Q_2 have four patterns, while Q_3 has only two.
    const int quads[3][4] =
    {
        { 1<<3                 ,          1<<2          ,                 1<<1   ,                       1<<0 },
        {     (1<<2)|(1<<1)|(1),   (1<<3)|    (1<<1)|(1),   (1<<3)|(1<<2)|    (1),   (1<<3)|(1<<2)|(1<<1)     },
        {     (1<<2)|(1<<1)    ,   (1<<3)|           (1),            /*unused*/-1,               /*unused*/-1 }
    };

    // masks to know if a pixel is accessible and if it has been already added to some region
    vector* accessible_pixel_mask = malloc(sizeof(vector)); //bool  
    vector_init_n(accessible_pixel_mask, width*height);

    vector* accumulated_pixel_mask = malloc(sizeof(vector)); //bool
    vector_init_n(accumulated_pixel_mask, width*height);

    // heap of boundary pixels
    vector boundary_pixes[256];
    vector boundary_edges[256];
    for(int i = 0;i < 256;i++)
    {
    	vector_init(&boundary_pixes[i]);
    	vector_init(&boundary_edges[i]);
    }

    // add a dummy-component before start
    ERStat* dummy = malloc(sizeof(ERStat));
    init_ERStat(dummy, 256, 0, 0, 0);
    vector_add(er_stack, &dummy);

    // we'll look initially for all pixels with grey-level lower than a grey-level higher than any allowed in the image
    int threshold_level = (255/filter->thresholdDelta)+1;

    // starting from the first pixel (0,0)
    int current_pixel = 0;
    int current_edge = 0;
    int current_level = image_data[0];
    bool val = true;
    vector_add(accessible_pixel_mask, &val);
    bool push_new_component = true;

    for(;;)
    {
    	int x = current_pixel % width;
        int y = current_pixel / width;

        // push a component with current level in the component stack
        if(push_new_component)
        {
        	ERStat* ers;
        	init_ERStat(ers, current_level, current_pixel, x, y);
        	vector_add(er_stack, &ers);
        }
        push_new_component = false;

        // explore the (remaining) edges to the neighbors to the current pixel
        for(; current_edge < 4; current_edge++)
        {

        	int neighbour_pixel = current_pixel;

        	switch (current_edge)
            {
                    case 0: if (x < width - 1) neighbour_pixel = current_pixel + 1;  break;
                    case 1: if (y < height - 1) neighbour_pixel = current_pixel + width; break;
                    case 2: if (x > 0) neighbour_pixel = current_pixel - 1; break;
                    default: if (y > 0) neighbour_pixel = current_pixel - width; break;
            }

            // if neighbour is not accessible, mark it accessible and retrieve its grey-level value
            if(! *(bool*)vector_get(accessible_pixel_mask, neighbour_pixel) && (neighbour_pixel != current_pixel))
            {

                int neighbour_level = image_data[neighbour_pixel];
                vector_set(accessible_pixel_mask, neighbour_pixel, &val);

                // if neighbour level is not lower than current level add neighbour to the boundary heap
                if(neighbour_level >= current_level)
                {
                	int curr_edge = 0;
                    vector_add(&boundary_pixes[neighbour_level], &neighbour_pixel);
                    vector_add(&boundary_edges[neighbour_level], &curr_edge);

                    // if neighbour level is lower than our threshold_level set threshold_level to neighbour level
                    if (neighbour_level < threshold_level)
                        threshold_level = neighbour_level;

                }
                else // if neighbour level is lower than current add current_pixel (and next edge)
                     // to the boundary heap for later processing
                {
                	int curr_edge = current_edge + 1;
                    vector_add(&boundary_pixes[current_level], &current_pixel);
                    vector_add(&boundary_edges[current_level], &curr_edge);

                    // if neighbour level is lower than threshold_level set threshold_level to neighbour level
                    if (current_level < threshold_level)
                        threshold_level = current_level;

                    // consider the new pixel and its grey-level as current pixel
                    current_pixel = neighbour_pixel;
                    current_edge = 0;
                    current_level = neighbour_level;

                    // and push a new component
                    push_new_component = true;
                    break;
                }
            }

        } // else neighbour was already accessible

        if (push_new_component) continue;

        // once here we can add the current pixel to the component at the top of the stack
        // but first we find how many of its neighbours are part of the region boundary (needed for
        // perimeter and crossings calc.) and the increment in quads counts for Euler's number calc.
        int non_boundary_neighbours = 0;
        int non_boundary_neighbours_horiz = 0;

        int quad_before[4] = {0,0,0,0};
        int quad_after[4] = {0,0,0,0};
        quad_after[0] = 1<<1;
        quad_after[1] = 1<<3;
        quad_after[2] = 1<<2;
        quad_after[3] = 1;

        for(int edge = 0; edge < 8; edge++)
        {
            int neighbour4 = -1;
            int neighbour8 = -1;
            int cell = 0;
            switch(edge)
            {
                    case 0: if (x < width - 1) { neighbour4 = neighbour8 = current_pixel + 1;} cell = 5; break;
                    case 1: if ((x < width - 1)&&(y < height - 1)) { neighbour8 = current_pixel + 1 + width;} cell = 8; break;
                    case 2: if (y < height - 1) { neighbour4 = neighbour8 = current_pixel + width;} cell = 7; break;
                    case 3: if ((x > 0)&&(y < height - 1)) { neighbour8 = current_pixel - 1 + width;} cell = 6; break;
                    case 4: if (x > 0) { neighbour4 = neighbour8 = current_pixel - 1;} cell = 3; break;
                    case 5: if ((x > 0)&&(y > 0)) { neighbour8 = current_pixel - 1 - width;} cell = 0; break;
                    case 6: if (y > 0) { neighbour4 = neighbour8 = current_pixel - width;} cell = 1; break;
                    default: if ((x < width - 1)&&(y > 0)) { neighbour8 = current_pixel + 1 - width;} cell = 2; break;
            }
            if((neighbour4 != -1)&&(*(bool *)vector_get(accumulated_pixel_mask, neighbour4))&&(image_data[neighbour4]<=image_data[current_pixel]))
            {
                non_boundary_neighbours++;
                if ((edge == 0) || (edge == 4))
                    non_boundary_neighbours_horiz++;
            }

            int pix_value = image_data[current_pixel] + 1;
            if (neighbour8 != -1)
            {
                if (accumulated_pixel_mask[neighbour8])
                    pix_value = image_data[neighbour8];
            }

            if (pix_value<=image_data[current_pixel])
            {
                switch(cell)
                {
                    case 0:
                        quad_before[3] = quad_before[3] | (1<<3);
                        quad_after[3]  = quad_after[3]  | (1<<3);
                        break;
                    case 1:
                        quad_before[3] = quad_before[3] | (1<<2);
                        quad_after[3]  = quad_after[3]  | (1<<2);
                        quad_before[0] = quad_before[0] | (1<<3);
                        quad_after[0]  = quad_after[0]  | (1<<3);
                        break;
                    case 2:
                        quad_before[0] = quad_before[0] | (1<<2);
                        quad_after[0]  = quad_after[0]  | (1<<2);
                        break;
                    case 3:
                        quad_before[3] = quad_before[3] | (1<<1);
                        quad_after[3]  = quad_after[3]  | (1<<1);
                        quad_before[2] = quad_before[2] | (1<<3);
                        quad_after[2]  = quad_after[2]  | (1<<3);
                        break;
                    case 5:
                        quad_before[0] = quad_before[0] | (1);
                        quad_after[0]  = quad_after[0]  | (1);
                        quad_before[1] = quad_before[1] | (1<<2);
                        quad_after[1]  = quad_after[1]  | (1<<2);
                        break;
                    case 6:
                        quad_before[2] = quad_before[2] | (1<<1);
                        quad_after[2]  = quad_after[2]  | (1<<1);
                        break;
                    case 7:
                        quad_before[2] = quad_before[2] | (1);
                        quad_after[2]  = quad_after[2]  | (1);
                        quad_before[1] = quad_before[1] | (1<<1);
                        quad_after[1]  = quad_after[1]  | (1<<1);
                        break;
                    default:
                        quad_before[1] = quad_before[1] | (1);
                        quad_after[1]  = quad_after[1]  | (1);
                        break;
                }
            }
        }

        int C_before[3] = {0, 0, 0};
        int C_after[3] = {0, 0, 0};

        for(int p=0; p<3; p++)
        {
            for(int q=0; q<4; q++)
            {
                if((quad_before[0] == quads[p][q]) && ((p<2)||(q<2)) )
                    C_before[p]++;
                if((quad_before[1] == quads[p][q]) && ((p<2)||(q<2)) )
                    C_before[p]++;
                if((quad_before[2] == quads[p][q]) && ((p<2)||(q<2)) )
                    C_before[p]++;
                if((quad_before[3] == quads[p][q]) && ((p<2)||(q<2)) )
                    C_before[p]++;

                if((quad_after[0] == quads[p][q]) && ((p<2)||(q<2)) )
                    C_after[p]++;
                if((quad_after[1] == quads[p][q]) && ((p<2)||(q<2)) )
                    C_after[p]++;
                if((quad_after[2] == quads[p][q]) && ((p<2)||(q<2)) )
                    C_after[p]++;
                if((quad_after[3] == quads[p][q]) && ((p<2)||(q<2)) )
                    C_after[p]++;
            }
        }

        int d_C1 = C_after[0]-C_before[0];
        int d_C2 = C_after[1]-C_before[1];
        int d_C3 = C_after[2]-C_before[2];

        er_add_pixel(filter, *(ERStat**)vector_back(er_stack), x, y, non_boundary_neighbours, non_boundary_neighbours_horiz, d_C1, d_C2, d_C3);
        vector_set(&accumulated_pixel_mask, current_pixel, &val);

        // if we have processed all the possible threshold levels (the hea is empty) we are done!
        if(threshold_level == (255/filter->thresholdDelta)+1)
        {
        	// save the extracted regions into the output vector
            if(filter->regions->capacity < filter->num_accepted_regions+1)
                vector_resize(filter->resize, filter->num_accepted_regions+1);
            er_save(filter, *(ERStat**)vector_back(er_stack), NULL, NULL);

            // clean memory
            for(int r = 0; r < vector_size(er_stack);r++)
            {
            	ERStat* stat = *(ERStat**)vector_get(er_stack, i);
            	if(stat->crossings)
            		vector_free(stat->crossings);

            	deleteERStatTree(stat);
            }
            vector_free(er_stack);

            return;
        }


        // pop the heap of boundary pixels
        current_pixel = *(int*)vector_back(&boundary_pixes[threshold_level]);
        vector_delete(&boundary_pixes[threshold_level], vector_size(&boundary_pixes[threshold_level])-1);
        current_edge = *(int*)vector_back(&boundary_edges[threshold_level]);
        vector_delete(&boundary_edges[threshold_level], vector_size(&boundary_edges[threshold_level])-1);

        for(; threshold_level < (255/filter->thresholdDelta)+1; threshold_level++)
        {
            if (!vector_empty(&boundary_pixes[threshold_level]))
                break;
        }

        int new_level = image_data[current_pixel];

        // if the new pixel has higher grey value than the current one
        if(new_level != current_level)
        {

        	current_level = new_level;

        	// process components on the top of the stack until we reach the higher grey-level
        	while (*(ERStat**)vector_back(er_stack)->level < new_level)
        	{
        		ERStat* er = *(ERStat**)vector_back(er_stack);
        		vector_delete(er_stack, vector_size(er_stack)-1);

        		if(new_level < (*(ERStat**)vector_back(er_stack))->level)
        		{
        			ERStat* temp;
        			init_ERStat(temp, new_level, current_pixel, current_pixel%width, current_pixel/width);
        			er_merge(filter, *(ERStat**)vector_back(er_stack), er);
        			break;
        		}

        		er_merge(*(ERStat**)vector_back(er_stack), er);
        	}

        }
    }
}

// accumulate a pixel into an ER
void er_add_pixel(ERFilterNM* filter, ERStat* parent, int x, int y, int non_border_neighbours,
                                                            int non_border_neighbours_horiz,
                                                            int d_C1, int d_C2, int d_C3)
{
    int add_val;
	parent->area++;
    parent->perimeter += 4 - 2*non_border_neighbours;
    if(vector_size(parent->crossings) > 0)
    {
    	if(y < parent->(rect.y))
        {
            add_val = 2;
    		vector_addfront(parent->crossings, &add_val);
        }

    	else if(y > (parent->rect).y + rect.height - 1)
        {
            add_val = 2;
    		vector_add(parent->crossings, &add_val);
        }

    	else
        {
            add_val = *(int*)vector_get(parent->crossings, y - parent->rect.y) + 2-2*non_border_neighbours_horiz;
    		vector_set(parent->crossings, y - parent->rect.y, &add_val);
        }
    }
    else
    {
        add_val = 2;
    	vector_add(parent->crossings, &add_val);
    }

    parent->euler += (d_C1 - d_C2 + 2*d_C3) / 4;

    int new_x1 = min(parent->rect.x, x);
    int new_y1 = min(parent->rect.y, y);
    int new_x1 = max(br(parent->rect).x -1, x);
    int new_y2 = max(br(parent->rect).y -1, y);
    parent->rect.x = new_x1;
    parent->rect.y = new_y1;
    parent->rect.width  = new_x2-new_x1+1;
    parent->rect.height = new_y2-new_y1+1;

    parent->raw_moments[0] += x;
    parent->raw_moments[1] += y;

    parent->central_moments[0] += x * x;
    parent->central_moments[1] += x * y;
    parent->central_moments[2] += y * y;
}

// merge an ER with its nested parent
void er_merge(ERFilterNM* filter, ERStat* parent, ERStat* child)
{
    int add_val;
	parent->area += child->area;

	parent->perimeter += child->perimeter;

	for(int i = parent->rect.y; i <= min(br(parent->rect).y - 1, br(child->rect).y-1); i++)
    {
        if(i-child->rect.y >= 0)
        {
            add_val = *(int*)vector_get(parent->crossings, i-parent->rect.y) + *(int *)vector_get(child->crossings, i-child->rect.y)
        	vector_set(parent->crossings, i-parent->rect.y, &add_val);
        }
    }

    for(int i = parent->rect.y-1; i >= child->rect.y; i--)
    {
        if (i-child->rect.y < vector_size(child->crossings));
        {
            add_val = *(int*)vector_get(child->crossings, i-child->rect.y)
            vector_add(parent->crossings, &add_val);
        }
        else
        {
        	add_val = 0;
            vector_addfront(parent->crossings, &add_val);
        }
    }

    for(int i = br(parent->rect).y; i < child->rect.y; i++)
    {
        add_val = 0;
    	vector_add(parent->crossings, &add_val);
    }

    for(int i = max(br(parent->rect).y, child->rect.y); i <= br(child->rect).y-1; i++)
    {
        add_val = *(int*)vector_get(child->crossings, i-child->rect.y)
    	vector_add(parent_crossings, &add_val);
    }

    parent->euler += child->euler;

    int new_x1 = min(parent->rect.x,child->rect.x);
    int new_y1 = min(parent->rect.y,child->rect.y);
    int new_x2 = max(br(parent->rect).x-1, br(child->rect).x-1);
    int new_y2 = max(br(parent->rect).y-1, br(child->rect).y-1);
	parent->rect.x = new_x1;
    parent->rect.y = new_y1;
    parent->rect.width  = new_x2-new_x1+1;
    parent->rect.height = new_y2-new_y1+1;

    parent->raw_moments[0] += child->raw_moments[0];
    parent->raw_moments[1] += child->raw_moments[1];

    parent->central_moments[0] += child->central_moments[0];
    parent->central_moments[1] += child->central_moments[1];
    parent->central_moments[2] += child->central_moments[2];

    vector m_crossings;
    vector_init(&m_crossings);
    add_val = *(int*)vector_get(child->crossings, child->rect.height/6);
    vector_add(&m_crossings, &add_val);
    add_val = *(int *)vector_get(child->crossings, 3*child->rect.height/6);
    vector_add(&m_crossings, &add_val);
    add_val = *(int *)vector_get(child->crossings, 5*child->rect.height/6)
    vector_add(&m_crossings, &add_val);
    sort_3ints(&m_crossings);
    child->med_crossings = (float)(*(int *)vector_get(&m_crossings, 1));

    // free unnecessary mem
    vector_free(child->crossings);

    // recover the original grey-level
    child->level = child->level*filter->thresholdDelta;

    // before saving calculate P(child|character) and filter if possible
    if(filter->classifier != NULL)
    {
        if(filter->stage == 1)
    	   child->probability = evalNM1(filter->classifier, *child);

        else if(filter->stage = 2)
            child->probability = evalNM2(filter->classifier, *child);

        else
            child->probability = eval_dummy(filter->classifier, *child);
    }

    if((((filter->classifier!=NULL)?(child->probability >= filter->minProbability):true)||(filter->nonMaxSuppression)) &&
         ((child->area >= (filter->minArea*filter->region_mask.rows*filter->region_mask.cols)) &&
          (child->area <= (filter->maxArea*filter->region_mask.rows*filter->region_mask.cols)) &&
          (child->rect.width > 2) && (child->rect.height > 2)))
    {
    	filter->num_accepted_regions++;

        child->next = parent->child;
        if(parent->child)
            parent->child->prev = child;

        parent->child = child;
        child->parent = parent;
    }
    else
    {
    	filter->num_rejected_regions++;

        if(child->prev != NULL)
            child->prev->next = child->next;

        ERStat* new_child = child->child;
        if(new_child != NULL)
        {
            while (new_child->next != NULL)
                new_child = new_child->next;

            new_child->next = parent->child;
            if (parent->child)
                parent->child->prev = new_child;

            parent->child   = child->child;
            child->child->parent = parent;
        }

        // free mem
        if(child->crossings)
            vector_free(child->crossings);
        free(child);
	}
}

// copy extracted regions into the output vector
ERStat* er_save(ERFilterNM* filter, ERStat *er, ERStat *parent, ERStat *prev)
{
	vector_add(filter->regions, *er);

	((ERStat*)vector_back(regions))->parent = parent;
	if(prev != NULL)
		prev->next = (ERStat*)vector_back(regions);
	
	else if(parent != NULL)
		parent->child = (ERStat*)vector_back(regions);

	ERStat *old_prev = NULL;
	ERStat *this_er  = (ERStat*)vector_back(regions);

	if(this_er->parent == NULL)
    {
       this_er->probability = 0;
    }

    if(filter->nonMaxSuppression)
    {
    	if(this_er->parent == NULL)
    	{
    		this_er->max_probability_ancestor = this_er;
            this_er->min_probability_ancestor = this_er;
    	}
    	else
    	{
    		this_er->max_probability_ancestor = (this_er->probability > parent->max_probability_ancestor->probability)? this_er :  parent->max_probability_ancestor;

    		this_er->min_probability_ancestor = (this_er->probability < parent->min_probability_ancestor->probability)? this_er :  parent->min_probability_ancestor;

    		if ((this_er->max_probability_ancestor->probability > filter->minProbability) && (this_er->max_probability_ancestor->probability - this_er->min_probability_ancestor->probability > filter->minProbabilityDiff))
    		{
    			this_er->max_probability_ancestor->local_maxima = true;
    			if ((this_er->max_probability_ancestor == this_er) && (this_er->parent->local_maxima))
    			    this_er->parent->local_maxima = false;
    		}
    		else if (this_er->probability < this_er->parent->probability)
            {
              this_er->min_probability_ancestor = this_er;
            }
            else if (this_er->probability > this_er->parent->probability)
            {
              this_er->max_probability_ancestor = this_er;
            }
        }
    }
    for(ERStat* child = er->child; child; child = child->next)
    {
    	old_prev = er_save(filter, child, this_er, old_prev);
    }
    return this_er;
}

// recursively walk the tree and filter (remove) regions using the callback classifier
ERStat* er_tree_filter(ERFilterNM* filter, Mat* src, ERStat* stat, ERStat *parent, ERStat *prev)
{
	assert(type(*src) == CV_8UC1);

	//Fill the region and calculate 2nd stage features
    Mat region = createusingRect(filter->region_mask, createRect(tl(stat->rect), init_Point(br(stat->rect).x + 2, br(stat->rect).y + 2)));
    createusingScalar(&region, init_Scalar(0, 0, 0, 0));

    int newMaskVal = 255;
    int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
    Rect rect;

    *src = createusingRect(*src, stat->rect);
    floodFill(src, &region, init_Point(stat->pixel%src->cols - stat->rect.x, stat->pixel/src->cols - stat->rect.y),
               init_Scalar(255, 0, 0, 0), &rect, init_Scalar(stat->level, 0, 0, 0), init_Scalar(0, 0, 0, 0), flags);

    region = createusingRect(region, init_Rect(1, 1, rect.width, rect.height));

    vector** contours;    /* Point */ //uninitialized, take care of this.
    vector* contour_poly = malloc(sizeof(vector)); /* Point */
    vector_init(contour_poly);
    vector* hierarchy = malloc(sizeof(vector));    /* Scalar */
    vector_init(hierarchy);

    findContours(region, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, init_Point(0, 0));
    //TODO check epsilon parameter of approxPolyDP (set empirically) : we want more precision
    //     if the region is very small because otherwise we'll loose all the convexities
    approxPolyDP(Mat(contours[0]), contour_poly, (float)min(rect.width,rect.height)/17, true);

    bool was_convex = false;
    int  num_inflexion_points = 0;

    for(int p = 0 ; p<(int)vector_size(contour_poly); p++)
    {
        int p_prev = p-1;
        int p_next = p+1;
        if(p_prev == -1)
            p_prev = (int)vector_size(contour_poly)-1;
        if(p_next == (int)vector_size(contour_poly))
            p_next = 0;

        double angle_next = atan2((double)(contour_poly[p_next].y-contour_poly[p].y),
                                  (double)(contour_poly[p_next].x-contour_poly[p].x));
        double angle_prev = atan2((double)(contour_poly[p_prev].y-contour_poly[p].y),
                                  (double)(contour_poly[p_prev].x-contour_poly[p].x));

        if(angle_next < 0)
            angle_next = 2.*CV_PI + angle_next;

        double angle = (angle_next - angle_prev);
        if(angle > 2.*CV_PI)
            angle = angle - 2.*CV_PI;
        else if(angle < 0)
            angle = 2.*CV_PI + fabs(angle);

        if(p>0)
        {
            if(((angle > CV_PI) && (!was_convex)) || ((angle < CV_PI) && (was_convex)))
                num_inflexion_points++;
        }
        was_convex = (angle > CV_PI);
    }

    Mat* m;
    m->flags = MAGIC_VAL;
    m->rows = m->cols = 0;
    m->data = 0;
    m->datastart = 0;
    m->dataend = 0;
    m->datalimit = 0;
    m->step = 0;
    floodFill(&region,  m, init_Point(0,0), init_Scalar(255, 0, 0, 0), 0, init_Scalar(0, 0, 0, 0), init_Scalar(0, 0, 0, 0), 4);
    int holes_area = region.cols*region.rows-countNonZero(region);

    int hull_area = 0;
    
    {
        vector* hull = malloc(sizeof(vector));
        vector_init(hull);
        convexHull(contours[0], hull, false);
        hull_area = (int)contourArea(hull);
    }

    stat->hole_area_ratio = (float)holes_area / stat->area;
    stat->convex_hull_ratio = (float)hull_area / (float)contourArea(contours[0]);
    stat->num_inflexion_points = (float)num_inflexion_points;

    // calculate P(child|character) and filter if possible
    if ((filter->classifier != NULL) && (stat->parent != NULL) )
    {
        if(filter->stage == 1)
            stat->probability = evalNM1(filter->classifier, *stat);
        else
            stat->probability = evalNM2(filter->classifier, *stat);
    }

    if((((filter->classifier != NULL)?(stat->probability >= filter->minProbability):true) &&
          ((stat->area >= filter->minArea*filter->region_mask.rows*filter->region_mask.cols) &&
           (stat->area <= filter->maxArea*filter->region_mask.rows*filter->region_mask.cols))) ||
        (stat->parent == NULL))
    {
        filter->num_accepted_regions++;
        vector_add(filter->regions, *stat);
        
        ((ERStat*)vector_back(regions))->parent = parent;
        ((ERStat*)vector_back(regions))->next   = NULL;
        ((ERStat*)vector_back(regions))->child  = NULL;

        if (prev != NULL)
            prev->next = ((ERStat*)vector_back(regions));
        else if (parent != NULL)
            parent->child = ((ERStat*)vector_back(regions));

        ERStat *old_prev = NULL;
        ERStat *this_er  = ((ERStat*)vector_back(regions));

        for (ERStat* child = stat->child; child; child = child->next)
        {
            old_prev = er_tree_filter(filter, src, child, this_er, old_prev);
        }

        return this_er;
    }

    else
    {
        num_rejected_regions++;

        ERStat* old_prev = prev;

        for(ERStat* child = stat->child; child; child = child->next)
        {
            old_prev = er_tree_filter(filter, src, child, parent, old_prev);
        }

        return old_prev;
    }
}

// recursively walk the tree selecting only regions with local maxima probability
ERStat* er_tree_nonmax_suppression(ERFilterNM* filter, ERStat* stat, ERStat* parent, ERStat* prev)
{
    if(stat->local_maxima || stat->parent == NULL)
    {
        vector_add(filter->regions, stat);

        ((ERStat*)vector_back(filter->regions))->parent = parent;
        ((ERStat*)vector_back(filter->regions))->next = NULL;
        ((ERStat*)vector_back(filter->regions))->child = NULL;

        if(prev != NULL)
            prev->next = (ERStat*)vector_back(filter->regions);

        else if(parent != NULL)
            parent->child = (ERStat*)vector_back(filter->regions);

        ERStat* old_prev = NULL;
        ERStat* this_er  = (ERStat*)vector_back(regions);

        for (ERStat* child = stat->child; child; child = child->next)
        {
            old_prev = er_tree_nonmax_suppression(filter, child, this_er, old_prev);
        }

        return this_er;
    }
    else
    {
        filter->num_rejected_regions++;
        filter->num_accepted_regions--;

        ERStat *old_prev = prev;

        for (ERStat* child = stat->child; child; child = child->next)
        {
            old_prev = er_tree_nonmax_suppression(filter, child, parent, old_prev);
        }

        return old_prev;
    }
}

static void deleteERStatTree(ERStat* root)
{
    vector* to_delete = malloc(sizeof(vector)); //ERStat*
    vector_init(to_delete);
    vector_add(root);
    while(!vector_empty(&to_delete))
    {
        ERStat* n = *(ERStat**)vector_front(to_delete);
        vector_delete(to_delete, 0);
        ERStat* c = n->child;
        if(c != NULL)
        {
            vector_add(to_delete, &c);
            ERStat* sibling = c->next;
            while(sibling != NULL)
            {
                vector_add(to_delete, &sibling);
                sibling = sibling->next;
            }
        }
        free(n);
    }
}


// Calculates the distance between two line estimates defined as the largest
// normalized vertical difference of their top/bottom lines at their boundary points
// out float distance
float distanceLinesEstimates(line_estimates a, line_estimates b)
{
    assert(a.h_max != 0 && b.h_max != 0);

    if(equal_line_estimates(a, b))
        return 0.0f;

    int x_min = min(a.x_min, b.x_min);
    int x_max = max(a.x_max, b.x_max);
    int h_max = max(a.h_max, b.h_max);

    float dist_top = FLT_MAX, dist_bottom = FLT_MAX;
    for(int i = 0; i < 2; i++)
    {
        float top_a0, top_a1, bottom_a0, bottom_a1;
        if(i == 0)
        {
            top_a0 = a.top1_a0;
            top_a1 = a.top1_a1;
            bottom_a0 = a.bottom1_a0;
            bottom_a1 = a.bottom1_a1;
        }
        else
        {
            top_a0 = a.top2_a0;
            top_a1 = a.top2_a1;
            bottom_a0 = a.bottom2_a0;
            bottom_a1 = a.bottom2_a1;
        }

        for(int j = 0; j < 2; j++)
        {
            float top_b0, top_b1, bottom_b0, bottom_b1;
            if (j==0)
            {
                top_b0 = b.top1_a0;
                top_b1 = b.top1_a1;
                bottom_b0 = b.bottom1_a0;
                bottom_b1 = b.bottom1_a1;
            } 

            else
            {
                top_b0 = b.top2_a0;
                top_b1 = b.top2_a1;
                bottom_b0 = b.bottom2_a0;
                bottom_b1 = b.bottom2_a1;
            }

            float x_min_dist = abs((top_a0+x_min*top_a1)-(top_b0+x_min*top_b1));
            float x_max_dist = abs((top_a0+x_max*top_a1)-(top_b0+x_max*top_b1));
            dist_top = min(dist_top, max(x_min_dist,x_max_dist)/h_max);
        }
    }
    return max(dist_top, dist_bottom);
}


// Evaluates if a set of more than 3 regions is valid or not
// using thresholds learned on training (defined above)
bool isValidSequence(region_sequence sequence1, region_sequence sequence2)
{
    for(size_t i = 0; i < vector_size(sequence2.triplets); i++)
    {
        for(size_t j = 0; j < vector_size(sequence1.triplets); j++)
        {
            if ((distanceLinesEstimates((region_triplet*)vector_get(sequence2.triplets, i)->estimates,
                                       (region_triplet*)vector_get(sequence1.triplets, j)->estimates) < SEQUENCE_MAX_TRIPLET_DIST) &&
                ((float)max(((region_triplet*)vector_get(sequence2.triplets, i)->estimates.x_min-(region_triplet*)vector_get(sequence1.triplets, j)->estimates.x_max),
                            ((region_triplet*)vector_get(sequence1.triplets, j)->estimates.x_min-(region_triplet*)vector_get(sequence2.triplets, i)->estimates.x_max))/
                        max((region_triplet*)vector_get(sequence2.triplets, i)->estimates.h_max,(region_triplet*)vector_get(sequence1.triplets, j)->estimates.h_max) < 3*PAIR_MAX_REGION_DIST))
                return true;
        }
    }

    return false;
}

bool vector_contains(vector* v, Point a) //Checks specifically for the struct Point
{
    for(int i = 0; i < vector_size(v); i++)
    {
        Point pt = *(Point*)vector_get(v, i);
        if(pt.x == a.x && pt.y == a.y)
            return true;
    }
    return false;
}

void erGroupingNM(Mat img, vector* src/* Mat */, vector** regions /* ERStat */, vector** out_groups /* Point */, vector* out_boxes /* Rect */, bool do_feedback_loop)
{
    assert(!vector_empty(src));
    size_t num_channels = vector_size(src);

    //process each channel independently
    for(size_t c = 0; c < num_channels; c++)
    {
        //store indices to regions in a single vector
        vector* all_regions = malloc(sizeof(vector)); //Point
        vector_init(all_regions);
        for(size_t r = 0; r < vector_size(regions[c]); r++)
        {
            Point push_pt = init_Point((int)c, (int)r)
            vector_add(all_regions, &push_pt);
        }

        vector* valid_pairs = malloc(sizeof(vector)); //region_pairs
        vector_init(valid_pairs);

        Mat* mask = malloc(sizeof(mat));
        zeros(mask, img.rows+2, img.cols+2, CV_8UC1);
        Mat grey, lab;
        cvtColor(img, lab, COLOR_RGB2Lab);
        cvtColor(img, &grey, COLOR_RGB2GRAY);

        //check every possible pair of regions
        for(size_t i = 0; i < vector_size(all_regions); i++)
        {
            vector* i_siblings = malloc(sizeof(vector)); //int
            vector_init(i_siblings);
            int first_i_sibling_idx = vector_size(valid_pairs);
            for(size_t j = i+1; j < vector_size(all_regions); j++)
            {
                // check height ratio, centroid angle and region distance normalized by region width
                // fall within a given interval
                if (isValidPair(grey, lab, mask, src, regions, *(Point*)vector_get(all_regions, i), *(Point*)vector_get(all_regions, j)));
                {
                    bool isCycle = false;
                    for(size_t k = 0; k < vector_size(i_siblings_); k++)
                    {
                        if(isValidPair(grey, lab, mask, src, regions, *(Point*)vector_get(all_regions, j), *(Point*)vector_get(all_regions, *(int*)vector_get(i_siblings, k))))
                        {
                            // choose as sibling the closer and not the first that was "paired" with i
                            Point i_center = init_Point((ERStat*)vector_get(regions[(Point*)vector_get(all_regions, i)->x], ((Point*)vector_get(all_regions, i))->y)->rect.x +
                                                (ERStat*)vector_get(regions[(Point*)vector_get(all_regions, i)->x], ((Point*)vector_get(all_regions, i))->y)->rect.width/2,
                                                (ERStat*)vector_get(regions[(Point*)vector_get(all_regions, i)->x], ((Point*)vector_get(all_regions, i))->y)->rect.y +
                                                (ERStat*)vector_get(regions[(Point*)vector_get(all_regions, i)->x], ((Point*)vector_get(all_regions, i))->y)->rect.height/2);
                            Point j_center = init_Point((ERStat*)vector_get(regions[(Point*)vector_get(all_regions, j)->x], ((Point*)vector_get(all_regions, j))->y)->rect.x +
                                                (ERStat*)vector_get(regions[(Point*)vector_get(all_regions, j)->x], ((Point*)vector_get(all_regions, j))->y)->rect.width/2,
                                                (ERStat*)vector_get(regions[(Point*)vector_get(all_regions, j)->x], ((Point*)vector_get(all_regions, j))->y)->rect.y +
                                                (ERStat*)vector_get(regions[(Point*)vector_get(all_regions, j)->x], ((Point*)vector_get(all_regions, j))->y)->rect.height/2);
                            Point k_center = init_Point( (ERStat*)vector_get(regions[(Point*)vector_get(all_regions, k)->x], ((Point*)vector_get(all_regions, k))->y)->rect.x +
                                                (ERStat*)vector_get(regions[(Point*)vector_get(all_regions, k)->x], ((Point*)vector_get(all_regions, k))->y)->rect.width/2,
                                                (ERStat*)vector_get(regions[(Point*)vector_get(all_regions, k)->x], ((Point*)vector_get(all_regions, k))->y)->rect.y +
                                                (ERStat*)vector_get(regions[(Point*)vector_get(all_regions, k)->x], ((Point*)vector_get(all_regions, k))->y)->rect.height/2);

                            if(norm(i_center.x-j_center.x, i_center.y-j_center.y) < norm(i_center.x-k_center.x, i_center.y-k_center.y))
                            {
                                vector_set(valid_pairs, first_i_sibling_idx+k, &init_region_pair(*(Point*)vector_get(all_regions, i), *(Point*)vector_get(all_regions, j)));
                                vector_set(i_siblings, k, &((int)j));
                            }
                            isCycle = true;
                            break;
                        }
                    }
                    if(!isCycle)
                    {
                        vector_add(valid_pairs, &init_region_pair(*(Point*)vector_get(all_regions, i), *(Point*)vector_get(all_regions, j)));
                        vector_add(i_siblings, &j);
                    }
                }
            }
        }

        vector* valid_triplets = malloc(sizeof(vector));// region_triplet
        vector_init(valid_triplets);

        //check every possible triplet of regions
        for(size_t i=0; i < vector_size(valid_pairs); i++)
        {
            for(size_t j=i+1; j < vector_size(valid_pairs); j++)
            {
                // check collinearity rules
                region_triplet valid_triplet = init_region_triplet(init_Point(0,0),init_Point(0,0), init_Point(0,0));
                if (isValidTriplet(regions, *(region_pair*)vector_get(valid_pairs, i) , *(region_pair*)vector_get(valid_pairs, j), &valid_triplet))
                    vector_add(valid_triplets, &valid_triplet);
            }
        }

        vector* valid_sequences = malloc(sizeof(vector)); //region_sequence
        vector_init(valid_sequences);
        vector* pending_sequences = malloc(sizeof(vector)); //region_sequence
        vector_init(pending_sequences);

        for(size_t i=0; i < vector_size(valid_triplets); i++)
        {
            vector_add(pending_sequences, &Region_sequence((region_triplet*)vector_get(valid_triplets, i)));
        }

        for(size_t i = 0; i < vector_size(pending_sequences); i++)
        {
            bool expanded = false;
            for(size_t j = i+1; j < vector_size(pending_sequences); j++)
            {
                if(isValidSequence(*(region_sequence*)vector_get(pending_sequences, i), *(region_sequence*)vector_get(pending_sequences, j)))
                {
                    expanded = true;
                    
                    for(int it = vector_size(((region_sequence*)vector_get(pending_sequences, j))->triplets)-1; it >= 0; it--)
                        vector_addfront(((region_sequence*)vector_get(pending_sequences, i))->triplets, vector_get(((region_sequence*)vector_get(pending_sequences, j))->triplets, it));
        
                    vector_delete(pending_sequences, j);
                    j--;
                }
            }
            if(expanded)
                vector_add(valid_sequences, (region_sequence*)vector_get(pending_sequences, i));
        }

        // remove a sequence if one its regions is already grouped within a longer seq
        for(size_t i=0; i < vector_size(valid_sequences); i++)
        {
            for(size_t j=i+1; j < vector_size(valid_sequences); j++)
            {
              if(haveCommonRegion(*(region_sequence*)vector_get(valid_sequences, i), *(region_sequence*)vector_get(valid_sequences, i)))
              {
                if(vector_size(((region_sequence*)vector_get(valid_sequences, i)->triplets)) < vector_size(((region_sequence*)vector_get(valid_sequences, j)->triplets)))
                {
                    vector_delete(valid_sequences, i);
                    i--;
                    break;
                }
                else
                {
                    vector_delete(valid_sequences, j);
                    j--;
                }
              }
            }
        }

        if(do_feedback_loop)
        {
            //Feedback loop of detected lines to region extraction ... tries to recover mismatches in the region decomposition step by extracting regions in the neighbourhood of a valid sequence and checking if they are consistent with its line estimates
            ERFilterNM* er_filter = createERFilterNM1(loadDummyClassifier(),1,0.005f,0.3f,0.f,false, 0.1f, true);
            for(int i = 0; i < (int)vector_size(valid_sequences); i++)
            {
                vector* bbox_points = malloc(sizeof(vector)); //Point
                vector_init(bbox_points);

                for(size_t j = 0; j < vector_size((*(region_sequence)vector_get(valid_sequences, i))->triplets); j++)
                {
                    vector_add(bbox_points, &tl(((ERStat*)vector_get(regions[((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->a.x], ((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->a.y))->rect));
                    vector_add(bbox_points, &br(((ERStat*)vector_get(regions[((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->a.x], ((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->a.y))->rect));
                    vector_add(bbox_points, &tl(((ERStat*)vector_get(regions[((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->b.x], ((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->b.y))->rect));
                    vector_add(bbox_points, &br(((ERStat*)vector_get(regions[((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->b.x], ((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->b.y))->rect));
                    vector_add(bbox_points, &tl(((ERStat*)vector_get(regions[((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->c.x], ((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->c.y))->rect));
                    vector_add(bbox_points, &br(((ERStat*)vector_get(regions[((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->c.x], ((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->c.y))->rect));
                }

                Rect rect = boundingRect(bbox_points);
                rect.x = max(rect.x-10,0);
                rect.y = max(rect.y-10,0);
                rect.width = min(rect.width+20, (*(Mat*)vector_get(src, c))->cols-rect.x);
                rect.height = min(rect.height+20, (*(Mat*)vector_get(src, c))->rows-rect.y);

                vector* aux_regions = malloc(sizeof(vector)); //ERStat
                vector_init(aux_regions);
                Mat tmp;
                copyTo(createusingRect(*(Mat*)vector_get(src, c), rect), &tmp);
                run(er_filter, tmp, aux_regions);

                for(size_t r = 0; r < vector_size(aux_regions); r++)
                {
                    if((((ERStat*)vector_get(aux_regions, r))->rect.y == 0) || (br(((ERStat*)vector_get(aux_regions, r))->rect).y >= tmp.rows))
                      continue;

                    ((ERStat*)vector_get(aux_regions, r))->rect.x = ((ERStat*)vector_get(aux_regions, r))->rect.x + rect.x;
                    ((ERStat*)vector_get(aux_regions, r))->rect.y = ((ERStat*)vector_get(aux_regions, r))->rect.y + rect.y;
                    bool overlaps = false;
                    for(size_t j = 0; j < vector_size((region_triplet*)vector_size(valid_sequences, i)->triplets); j++)
                    {
                        Rect minarearect_a = ((ERStat*)vector_get(regions[((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->a.x], ((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->a.y))->rect | ((ERStat*)vector_get(aux_regions, r))->rect;
                        Rect minarearect_b = ((ERStat*)vector_get(regions[((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->b.x], ((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->b.y))->rect | ((ERStat*)vector_get(aux_regions, r))->rect
                        Rect minarearect_c = ((ERStat*)vector_get(regions[((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->c.x], ((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->c.y))->rect | ((ERStat*)vector_get(aux_regions, r))->rect

                        // Overlapping regions are not valid pair in any case
                        if(equalRects(minarearect_a, ((ERStat*)vector_get(aux_regions, r))->rect) || 
                           equalRects(minarearect_b, ((ERStat*)vector_get(aux_regions, r))->rect) ||
                           equalRects(minarearect_c, ((ERStat*)vector_get(aux_regions, r))->rect) ||
                           equalRects(minarearect_a, ((ERStat*)vector_get(regions[((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->a.x], ((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->a.y))->rect) ||
                           equalRects(minarearect_b, ((ERStat*)vector_get(regions[((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->b.x], ((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->b.y))->rect) ||
                           equalRects(minarearect_c, ((ERStat*)vector_get(regions[((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->c.x], ((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->c.y))->rect))
                        {
                            overlaps = true;
                            break;
                        }
                    }
                    if(!overlaps)
                    {
                        //now check if it has at least one valid pair
                        vector *left_couples = malloc(sizeof(vector)), *right_couples = malloc(sizeof(vector)); //Vec3i
                        vector_init(left_couples);
                        vector_init(right_couples);
                        vector_add(regions[c], ((ERStat*)vector_get(aux_regions, r)));
                        for(size_t j=0; j < vector_size((region_triplet*)vector_get(valid_sequences, i)->triplets); j++)
                        {
                            if(isValidPair(grey, lab, mask, src, regions, ((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->a, init_Point(((int)c,(int)(vector_size(regions[c]))-1))))
                            {
                                if(((ERStat*)vector_get(regions[((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->a.x], ((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->a.y))->rect.x > (ERStat*)vector_get(aux_regions, r)->rect.x)
                                    vector_add(right_couples, vec3i(((ERStat*)vector_get(regions[((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->a.x], ((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->a.y))->rect.x - (ERStat*)vector_get(aux_regions, r)->rect.x, ((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->a.x, ((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->a.y));

                                else
                                    vector_add(left_couples, vec3i((ERStat*)vector_get(aux_regions, r)->rect.x - ((ERStat*)vector_get(regions[((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->a.x], ((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->a.y))->rect.x, ((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->a.x, ((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->a.y));
                            }
                            if(isValidPair(grey, lab, mask, src, regions, ((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->b, init_Point(((int)c,(int)(vector_size(regions[c]))-1))))
                            {
                                if(((ERStat*)vector_get(regions[((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->b.x], ((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->b.y))->rect.x > (ERStat*)vector_get(aux_regions, r)->rect.x)
                                    vector_add(right_couples, vec3i(((ERStat*)vector_get(regions[((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->b.x], ((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->b.y))->rect.x - (ERStat*)vector_get(aux_regions, r)->rect.x, ((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->b.x, ((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->b.y));

                                else
                                    vector_add(left_couples, vec3i((ERStat*)vector_get(aux_regions, r)->rect.x - ((ERStat*)vector_get(regions[((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->b.x], ((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->b.y))->rect.x, ((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->b.x, ((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->b.y));
                            }
                            if(isValidPair(grey, lab, mask, src, regions, ((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->c, init_Point(((int)c,(int)(vector_size(regions[c]))-1))))
                            {
                                if(((ERStat*)vector_get(regions[((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->c.x], ((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->c.y))->rect.x > (ERStat*)vector_get(aux_regions, r)->rect.x)
                                    vector_add(right_couples, vec3i(((ERStat*)vector_get(regions[((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->c.x], ((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->c.y))->rect.x - (ERStat*)vector_get(aux_regions, r)->rect.x, ((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->c.x, ((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->c.y));

                                else
                                    vector_add(left_couples, vec3i((ERStat*)vector_get(aux_regions, r)->rect.x - ((ERStat*)vector_get(regions[((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->c.x], ((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->c.y))->rect.x, ((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->c.x, ((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->c.y));
                            }                        
                        }

                        //make it part of a triplet and check if line estimates is consistent with the sequence
                        vector* new_valid_triplets = malloc(sizeof(vector)); //region_triplet
                        vector_init(new_valid_triplets);
                        if(!vector_empty(left_couples) && !vector_empty(right_couples))
                        {
                            sort(left_couples);
                            sort(right_couples);
                            region_pair pair1 = init_region_pair(init_Point((Vec3i*)vector_get(left_couples, 0)->val[1], (Vec3i*)vector_get(left_couples, 0)->val[2]), init_Point(c, vector_size(regions[c])-1));
                            region_pair pair2 = init_region_pair(init_Point(c, vector_size(regions[c])-1), init_Point((Vec3i*)vector_get(left_couples, 0)->val[1], (Vec3i*)vector_get(left_couples, 0)->val[2]));
                            region_triplet triplet = init_region_triplet(init_Point(0, 0), init_Point(0, 0), init_Point(0, 0));
                            
                            if(isValidTriplet(regions, pair1, pair2, &triplet))
                                vector_add(new_valid_triplets, &triplet);
                        }
                        else if(vector_size(right_couples) >= 2)
                        {
                            sort(right_couples);
                            region_pair pair1 = init_region_pair(init_Point(c, vector_size(regions[c])-1), init_Point((Vec3i*)vector_get(right_couples, 0)->val[1], (Vec3i*)vector_get(right_couples, 0)->val[2]));
                            region_pair pair2 = init_region_pair(init_Point((Vec3i*)vector_get(right_couples, 0)->val[1], (Vec3i*)vector_get(right_couples, 0)->val[2]), init_Point(init_Point((Vec3i*)vector_get(right_couples, 1)->val[1], (Vec3i*)vector_get(right_couples, 1)->val[2])))
                            region_triplet triplet = init_region_triplet(init_Point(0, 0), init_Point(0, 0), init_Point(0, 0));
                            
                            if(isValidTriplet(regions, pair1, pair2, &triplet))
                                vector_add(new_valid_triplets, &triplet);

                        }
                        else if(vector_size(left_couples) >= 2)
                        {
                            sort(left_couples);
                            region_pair pair1 = init_region_pair(init_Point((Vec3i*)vector_get(left_couples, 1)->val[1], (Vec3i*)vector_get(left_couples, 1)->val[2]), init_Point((Vec3i*)vector_get(left_couples, 0)->val[1], (Vec3i*)vector_get(left_couples, 0)->val[2]));
                            region_pair pair2 = init_region_pair(init_Point((Vec3i*)vector_get(left_couples, 0)->val[1], (Vec3i*)vector_get(left_couples, 0)->val[2]), init_Point(c, vector_size(regions[c])-1));
                            region_triplet triplet = init_region_triplet(init_Point(0, 0), init_Point(0, 0), init_Point(0, 0));
                            
                            if(isValidTriplet(regions, pair1, pair2, &triplet))
                                vector_add(new_valid_triplets, &triplet);
                        }
                        else
                        {
                            // no possible triplet found
                            continue;
                        }

                        //check if line estimates is consistent with the sequence
                        for (size_t t=0; t < vector_size(new_valid_triplets); t++)
                        {
                            region_sequence sequence = Region_sequence((region_triplet*)vector_get(new_valid_triplets, t));
                            if (isValidSequence(*(region_sequence*)vector_get(valid_sequences, i), sequence))
                                vector_add(((region_sequence*)vector_get(valid_sequences, i))->triplets, (region_triplet*)vector_get(new_valid_triplets, t));
                        }
                    }
                }
            }
        }


        // Prepare the sequences for output
        for(size_t i = 0; i < vector_size(valid_sequences); i++)
        {
            vector* bbox_points = malloc(sizeof(vector)); //Point
            vector_init(bbox_points);

            vector* group_regions = malloc(sizeof(vector)); //Point
            vector_init(group_regions);

            for(size_t j = 0; j < vector_size((region_sequence*)vector_get(valid_sequences, i)->triplets); j++)
            {
                size_t prev_size = vector_size(group_regions);

                if(!vector_contains(group_regions, (((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->a))) 
                    vector_add(group_regions, &(((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->a));

                if(!vector_contains(group_regions, (((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->b))) 
                    vector_add(group_regions, &(((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->b));

                if(!vector_contains(group_regions, (((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->c))) 
                    vector_add(group_regions, &(((region_triplet*)vector_get(((region_sequence*)vector_get(valid_sequences, i))->triplets, j))->c));

                for(size_t k=prev_size; k < vector_size(group_regions); k++)
                {
                    vector_add(bbox_points, &tl(((ERStat*)vector_get(regions[((Point*)vector_get(group_regions, k))->x], ((Point*)vector_get(group_regions, k))->y))->rect));
                    vector_add(bbox_points, &br(((ERStat*)vector_get(regions[((Point*)vector_get(group_regions, k))->x], ((Point*)vector_get(group_regions, k))->y))->rect));
                }
            }
            vector_add(out_groups, group_regions);
            vector_add(out_boxes, boundingRect(bbox_points));
        }

    }
}

// Evaluates if a pair of regions is valid or not
// using thresholds learned on training (defined above)
bool isValidPair(Mat grey, Mat lab, Mat mask, vector* channels, vector** regions, Point idx1, Point idx2)
{
    Rect minarearect = performOR(&((*(ERStat *)vector_get(regions[idx1.x], idx1[1])).rect) | &((*(ERStat *)vector_get(regions[idx2.x], idx2.y)).rect));

    // Overlapping regions are not valid pair in any case
    if(equalRects(minarearect, ((*(ERStat *)vector_get(regions[idx1.x], idx1.y)).rect)) || equalRects(minarearect, ((*(ERStat *)vector_get(regions[idx2.x], idx2.y)).rect)))
        return false;

    ERStat *i, *j;
    if(((*(ERStat*)vector_get(regions[idx1.x], idx1.y)).rect).x < ((*(ERStat *)vector_get(regions[idx2.x], idx2.y)).rect).x )
    {
        i = (ERStat*)vector_get(regions[idx1.x], idx1.y); //i = &regions[idx1[0]][idx1[1]];
        j = (ERStat*)vector_get(regions[idx2.x], idx2.y);
    }
    else
    {
        i = (ERStat*)vector_get(regions[idx2.x], idx2.y);
        j = (ERStat*)vector_get(regions[idx1.x], idx1.y);
    }

    if(j->rect.x == i->rect.x)
        return false;

    float height_ratio = (float)min(i->rect.height,j->rect.height) /
                                max(i->rect.height,j->rect.height);

    Point center_i, center_j;
    center_i = init_Point(i->rect.x+i->rect.width/2, i->rect.y+i->rect.height/2);
    center_j = init_Point(j->rect.x+j->rect.width/2, j->rect.y+j->rect.height/2);
    float centroid_angle = atan2f((float)(center_j.y-center_i.y), (float)(center_j.x-center_i.x));

    int avg_width = (i->rect.width + j->rect.width) / 2;
    float norm_distance = (float)(j->rect.x-(i->rect.x + i->rect.width))/avg_width;

    if ((height_ratio   < PAIR_MIN_HEIGHT_RATIO) ||
        (centroid_angle < PAIR_MIN_CENTROID_ANGLE) ||
        (centroid_angle > PAIR_MAX_CENTROID_ANGLE) ||
        (norm_distance  < PAIR_MIN_REGION_DIST) ||
        (norm_distance  > PAIR_MAX_REGION_DIST))
        return false;

    if ((i->parent == NULL) || (j->parent == NULL)) // deprecate the root region
      return false;

    i = (ERStat *)vector_get(regions[idx1.x], idx1.y);
    j = (ERStat *)vector_get(regions[idx2.x], idx2.y);

    Mat region = createusingRect(mask, createRect(init_Point(i->rect.x, i->rect.y), init_Point(i->rect.x + i->rect.width + 2, i->rect.y + i->rect.height + 2)));
    createusingScalar(&region, init_Scalar(0, 0, 0, 0));
    int newMaskVal = 255;
    int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;

    floodFill(&createusingRect(*(Mat*)vector_get(channels, idx1[0]), i->rect),
               &region, init_Point(i->pixel%grey.cols - i->rect.x, i->pixel/grey.cols - i->rect.y),
               init_Scalar(255, 0, 0, 0), NULL, init_Scalar(i->level, 0, 0, 0), init_Scalar(0, 0, 0, 0), flags);
    Mat rect_mask = createusingRect(mask, init_Rect(i->rect.x+1,i->rect.y+1,i->rect.width,i->rect.height));

    Scalar mean, std;
    meanStdDev(&createusingRect(grey, i->rect), &mean, &std, &rect_mask);
    int grey_mean1 = (int)mean.val[0];
    meanStdDev(&createusingRect(lab, i->rect), &mean, &std, &rect_mask);
    float a_mean1 = (float)mean.val[1];
    float b_mean1 = (float)mean.val[2];

    region = createusingRect(mask, createRect(init_Point(j->rect.x, j->rect.y), init_Point(j->rect.x + j->rect.width + 2, j->rect.y + j->rect.height + 2)));
    createusingScalar(&region, init_Scalar(0, 0, 0, 0));

    floodFill(&createusingRect(*(Mat*)vector_get(channels, idx2[0]), j->rect),
               &region, init_Point(j->pixel%grey.cols - j->rect.x, j->pixel/grey.cols - j->rect.y),
               init_Scalar(255, 0, 0, 0), NULL, init_Scalar(j->level, 0, 0, 0), init_Scalar(0, 0, 0, 0), flags);
    rect_mask = createusingRect(mask, init_Rect(j->rect.x+1,j->rect.y+1,j->rect.width,j->rect.height));

    meanStdDev(&createusingRect(grey, j->rect), &mean, &std, &rect_mask);
    int grey_mean2 = (int)mean.val[0];
    meanStdDev(&createusingRect(lab, i->rect), &mean, &std, &rect_mask);
    float a_mean2 = (float)mean.val[1];
    float b_mean2 = (float)mean.val[2];

    if(abs(grey_mean1-grey_mean2) > PAIR_MAX_INTENSITY_DIST)
      return false;

    if(sqrt(pow(a_mean1-a_mean2,2)+pow(b_mean1-b_mean2,2)) > PAIR_MAX_AB_DIST)
      return false;

    return true;
}


// Evaluates if a set of 3 regions is valid or not
// using thresholds learned on training (defined above)
bool isValidTriplet(vector** regions/* ERStat */, region_pair pair1, region_pair pair2, region_triplet* triplet/* output triplet */)
{
    if(equalRegionPairs(pair1, pair2))
        return false;

    // At least one region in common is needed
    if ((pair1.a.x == pair2.a.x && pair1.a.y == pair2.a.y) || 
        (pair1.a.x == pair2.b.x && pair1.a.y == pair2.b.y) || 
        (pair1.b.x == pair2.a.x && pair1.b.y == pair2.a.y) ||
        (pair1.b.x == pair2.b.x && pair1.b.y == pair2.b.y))
    {
         //fill the indices in the output triplet (sorted)
        if(pair1.a.x == pair2.a.x && pair1.a.y == pair2.a.y)
        {
            if(((*(ERStat*)vector_get(regions[pair1.b.x], pair1.b.y)).rect.x <= (*(ERStat*)vector_get(regions[pair1.a.x], pair1.a.y)).rect.x) &&
                    ((*(ERStat*)vector_get(regions[pair2.b.x], pair2.b.y)).rect.x <= (*(ERStat*)vector_get(regions[pair1.a.x], pair1.a.y)).rect.x))
                return false;

            if(((*(ERStat*)vector_get(regions[pair1.b.x], pair1.b.y)).rect.x >= (*(ERStat*)vector_get(regions[pair1.a.x], pair1.a.y)).rect.x) &&
                    ((*(ERStat*)vector_get(regions[pair2.b.x], pair2.b.y)).rect.x >= regions[pair1.a[0]][pair1.a[1]].rect.x))
                return false;

            triplet->a = ((*(ERStat*)vector_get(regions[pair1.b.x], pair1.b.y)).rect.x <
                         (*(ERStat*)vector_get(regions[pair2.b.x], pair2.b.y)).rect.x)? pair1.b : pair2.b;
            triplet->b = pair1.a;
            triplet->c = ((*(ERStat*)vector_get(regions[pair1.b.x], pair1.b.y)).rect.x >
                         (*(ERStat*)vector_get(regions[pair2.b.x], pair2.b.y)).rect.x)? pair1.b : pair2.b;
        }

        else if(pair1.a.x == pair2.b.x && pair1.a.y == pair2.b.y)
        {
            if(((*(ERStat*)vector_get(regions[pair1.b.x], pair1.b.y)).rect.x <= (*(ERStat*)vector_get(regions[pair1.a.x], pair1.a.y)).rect.x) &&
                    ((*(ERStat*)vector_get(regions[pair2.a.x], pair2.a.y)).rect.x <= (*(ERStat*)vector_get(regions[pair1.a.x], pair1.a.y)).rect.x))
                return false;

            if(((*(ERStat*)vector_get(regions[pair1.b.x], pair1.b.y)).rect.x >= (*(ERStat*)vector_get(regions[pair1.a.x], pair1.a.y)).rect.x) &&
                    ((*(ERStat*)vector_get(regions[pair2.a.x], pair2.a.y)).rect.x >= (*(ERStat*)vector_get(regions[pair1.a.x], pair1.a.y)).rect.x))
                return false;

            triplet->a = ((*(ERStat*)vector_get(regions[pair1.b.x], pair1.b.y)).rect.x <
                         (*(ERStat*)vector_get(regions[pair2.a.x], pair2.a.y)).rect.x)? pair1.b : pair2.a;
            triplet->b = pair1.a;
            triplet->c = ((*(ERStat*)vector_get(regions[pair1.b.x], pair1.b.y)).rect.x >
                         (*(ERStat*)vector_get(regions[pair2.a.x], pair2.a.y)).rect.x)? pair1.b : pair2.a;
        }

        else if(pair1.b.x == pair2.a.x && pair1.b.y == pair2.a.y)
        {
            if (((*(ERStat*)vector_get(regions[pair1.a.x], pair1.a.y)).rect.x <= (*(ERStat*)vector_get(regions[pair1.b.x], pair1.b.y)).rect.x) &&
                    ((*(ERStat*)vector_get(regions[pair2.b.x], pair2.b.y)).rect.x <= (*(ERStat*)vector_get(regions[pair1.b.x], pair1.b.y)).rect.x))
                return false;

            if (((*(ERStat*)vector_get(regions[pair1.a.x], pair1.a.y)).rect.x >= (*(ERStat*)vector_get(regions[pair1.b.x], pair1.b.y)).rect.x) &&
                    ((*(ERStat*)vector_get(regions[pair2.b.x], pair2.b.y)).rect.x >= (*(ERStat*)vector_get(regions[pair1.b.x], pair1.b.y)).rect.x))
                return false;

            triplet->a = ((*(ERStat*)vector_get(regions[pair1.a.x], pair1.a.y)).rect.x <
                         (*(ERStat*)vector_get(regions[pair2.b.x], pair2.b.y)).rect.x)? pair1.a : pair2.b;
            triplet->b = pair1.b;
            triplet->c = ((*(ERStat*)vector_get(regions[pair1.a.x], pair1.a.y)).rect.x <
                         (*(ERStat*)vector_get(regions[pair2.b.x], pair2.b.y)).rect.x)? pair1.a : pair2.b;
        }

        else if(pair1.b.x == pair2.b.x && pair1.b.y == pair2.b.y)
        {
            if(((*(ERStat*)vector_get(regions[pair1.a.x], pair1.a.y)).rect.x <= (*(ERStat*)vector_get(regions[pair1.b.x], pair1.b.y)).rect.x) &&
                    ((*(ERStat*)vector_get(regions[pair2.a.x], pair2.a.y)).rect.x <= (*(ERStat*)vector_get(regions[pair1.b.x], pair1.b.y)).rect.x))
                return false;

            if(((*(ERStat*)vector_get(regions[pair1.a.x], pair1.a.y)).rect.x >= (*(ERStat*)vector_get(regions[pair1.b.x], pair1.b.y)).rect.x) &&
                    ((*(ERStat*)vector_get(regions[pair2.a.x], pair2.a.y)).rect.x >= (*(ERStat*)vector_get(regions[pair1.b.x], pair1.b.y)).rect.x))
                return false;

            triplet->a = ((*(ERStat*)vector_get(regions[pair1.a.x], pair1.a.y)).rect.x <
                         (*(ERStat*)vector_get(regions[pair2.a.x], pair2.a.y)).rect.x)? pair1.a : pair2.a;
            triplet->b = pair1.b;
            triplet->c = ((*(ERStat*)vector_get(regions[pair1.a.x], pair1.a.y)).rect.x >
                         (*(ERStat*)vector_get(regions[pair2.a.x], pair2.a.y)).rect.x)? pair1.a : pair2.a;
        }

        if((((*ERStat*)vector_get(regions[triplet->a[0]], triplet->a[1])).rect.x == ((*ERStat*)vector_get(regions[triplet->b[0]], triplet->b[1])).rect.x) &&
             (((*ERStat*)vector_get(regions[triplet->a[0]], triplet->a[1])).rect.x == ((*ERStat*)vector_get(regions[triplet->c[0]], triplet->c[1])).rect.x))
            return false;

        if((((*ERStat*)vector_get(regions[triplet->a[0]], triplet->a[1])).rect.x + ((*ERStat*)vector_get(regions[triplet->a[0]], triplet->a[1])).rect.width
                 == ((*ERStat*)vector_get(regions[triplet->b[0]], triplet->b[1])).rect.x + ((*ERStat*)vector_get(regions[triplet->b[0]], triplet->b[1])).rect.width) &&
             (((*ERStat*)vector_get(regions[triplet->a[0]], triplet->a[1])).rect.x + ((*ERStat*)vector_get(regions[triplet->a[0]], triplet->a[1])).rect.width
                 == ((*ERStat*)vector_get(regions[triplet->c[0]], triplet->c[1])).rect.x + ((*ERStat*)vector_get(regions[triplet->c[0]], triplet->c[1])).rect.width) )
            return false;


        if (!fitLineEstimates(regions, triplet))
            return false;

        if((triplet->estimates.bottom1_a0 < triplet->estimates.top1_a0) ||
           (triplet->estimates.bottom1_a0 < triplet->estimates.top2_a0) ||
           (triplet->estimates.bottom2_a0 < triplet->estimates.top1_a0) ||
           (triplet->estimates.bottom2_a0 < triplet->estimates.top2_a0))
            return false;

        int central_height = (int)min(triplet->estimates.bottom1_a0, triplet->estimates.bottom2_a0) -
                             (int)max(triplet->estimates.top1_a0,triplet->estimates.top2_a0);
        int top_height     = (int)abs(triplet->estimates.top1_a0 - triplet->estimates.top2_a0);
        int bottom_height  = (int)abs(triplet->estimates.bottom1_a0 - triplet->estimates.bottom2_a0);

        if (central_height == 0)
            return false;

        float top_height_ratio    = (float)top_height/central_height;
        float bottom_height_ratio = (float)bottom_height/central_height;

        if((top_height_ratio > TRIPLET_MAX_DIST) || (bottom_height_ratio > TRIPLET_MAX_DIST))
            return false;

        if(abs(triplet->estimates.bottom1_a1) > TRIPLET_MAX_SLOPE)
            return false;

        return true;
    }

    return false;
}


// Fit line from three points using (heuristic) Least-Median of Squares
// out a0 is the intercept
// out a1 is the slope
// returns the error of the single point that doesn't fit the line
float fitLineLMS(Point p1, Point p2, Point p3, float* a0, float* a1)
{
    //if this is not changed the line is not valid
    *a0 = -1;
    *a1 = 0;

    //Least-Median of Squares does not make sense with only three points
    //because any line passing by two of them has median_error = 0
    //So we'll take the one with smaller slope
    float l_a0, l_a1, best_slope=FLT_MAX, err=0;

    if(p1.x != p2.x)
    {
        fitLine(p1,p2,l_a0,l_a1);
        if(abs(l_a1) < best_slope)
        {
            best_slope = abs(l_a1);
            a0 = l_a0;
            a1 = l_a1;
            err = (p3.y - (a0+a1*p3.x));
        }
    }

    if(p1.x != p3.x)
    {
        fitLine(p1,p3,l_a0,l_a1);
        if(abs(l_a1) < best_slope)
        {
            best_slope = abs(l_a1);
            a0 = l_a0;
            a1 = l_a1;
            err = (p2.y - (a0+a1*p2.x));
        }
    }

    if(p2.x != p3.x)
    {
        fitLine(p2,p3,l_a0,l_a1);
        if(abs(l_a1) < best_slope)
        {
            best_slope = abs(l_a1);
            a0 = l_a0;
            a1 = l_a1;
            err = (p1.y - (a0+a1*p1.x));
        }
    }

    return err;
}

void setThresholdDeta(ERFilterNM* filter, int thresholdDelta)
{
    filter->thresholdDelta = thresholdDelta;
}