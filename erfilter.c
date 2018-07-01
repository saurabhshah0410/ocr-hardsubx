#include <stdio.h>

/*!
    Compute the different channels to be processed independently in the N&M algorithm
    Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012

    In N&M algorithm, the combination of intensity (I), hue (H), saturation (S), and gradient
    magnitude channels (Grad) are used in order to obtain high localization recall.
    This implementation also the alternative combination of red (R), green (G), blue (B),
    lightness (L), and gradient magnitude (Grad).

    \param  _src           Source image. Must be RGB CV_8UC3.
    \param  _channels      Output vector<Mat> where computed channels are stored.
*/
void computeNMChannels(Mat src, vector* _channels/*output vector of Mat*/)
{
    if(empty(src)) {
        for(int i = 0;i < vector_size(_channels);i++)
        {
            Mat* m = vector_get(_channels, i);
            m->data = m->rows = m->cols = 0;
            m->step = 0;
        }
        vector_free(_channels);
        return;
    }

    assert(type(src) == CV_8UC3);
    createVector(_channels, 5, 1, src.depth(), -1, false, 0);

    vector* channelsRGB; /* Mat */
    split(src, channelsRGB);
    for (int i = 0; i < channels(src); i++)
    {
        create(_channels, src.rows, src.cols, CV_8UC1, i, false, 0);
        Mat channel = *(Mat*)(vector_get(channelsRGB, i));
        copyTo(*(Mat*)vector_get(channelsRGB, i), &channel);
    }

    Mat hls;
    cvtColor(src, hls, COLOR_RGB2HLS);
    vector* channelsHLS;/* Mat */
    split(hls, channelsHLS);

    create(_channels, src.rows, src.cols, CV_8UC1, 3, false, 0);
    Mat channelL =*(Mat*)(vector_get(channelsRGB, 3));
    copyTo(*(Mat*)vector_get(channelsRGB, 1), &channelL);

    Mat grey;
    cvtColor(src, grey, COLOR_RGB2GRAY, 0);
    Mat gradient_magnitude = Mat_<float>(grey.size());
    get_gradient_magnitude(grey, gradient_magnitude);
    convertTo(&gradient_magnitude, &gradient_magnitude, CV_8UC1, 1, 0);

    create(_channels, src.rows, src.cols, CV_8UC1, 4, false, 0);
    Mat channelGrad = *(Mat *)vector_get(_channels, 4);
    copyTo(gradient_magnitude, &channelGrad);
}

// The classifier must return probability measure for the region --> Stage-1
double evalNM1(ERClassifierNM erc, const ERStat& stat)
{
	float sample_[1][4] = {(float)(stat.rect.width)/(stat.rect.height), // aspect ratio
                     sqrt((float)(stat.area))/stat.perimeter, // compactness
                     (float)(1-stat.euler), //number of holes
                     stat.med_crossings};
	Mat sample;
	sample.rows = 1;
	sample.cols = 4;
	sample.flags = CV_64F;
	memcpy(sample.data, sample_, 1*4*sizeof(float)); //check this
	
	float votes = erc.boost->predict(sample, noArray(), PREDICT_SUM | RAW_OUTPUT);
	// PREDICT_SUM=(1<<8), RAW_OUTPUT=1.

	// Logistic Correction returns a probability value (in the range(0,1))
	return (double)1-(double)1/(1+exp(-2*votes));
}

// The classifier must return probability measure for the region --> Stage-2
double evalNM2(ERClassifierNM erc, const ERStat& stat)
{
	float sample_[1][7] = {(float)(stat.rect.width)/(stat.rect.height), // aspect ratio
                     sqrt((float)(stat.area))/stat.perimeter, // compactness
                     (float)(1-stat.euler), //number of holes
                     stat.med_crossings, stat.hole_area_ratio,
                     stat.convex_hull_ratio, stat.num_inflexion_points};
    Mat sample;
    sample.rows = 1;
	sample.cols = 7;
	sample.flags = CV_64F;
	memcpy(sample.data, sample_, 1*7*sizeof(float)); //check this
	float votes = erc.boost->predict(sample, noArray(), PREDICT_SUM | RAW_OUTPUT);

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
		erc->boost = load(filename);
		if(empty(erc->boost))
			Fatal("Could not read the Default classifier\n");
		return erc;
	}
	else
		Fatal("Default classifier file not found!");
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
                              default classifier can be implicitly load with function loadClassifierNM1()
                              from file in samples/cpp/trained_classifierNM1.xml
    \param  thresholdDelta    Threshold step in subsequent thresholds when extracting the component tree
    \param  minArea           The minimum area (% of image size) allowed for retrieved ER's
    \param  minArea           The maximum area (% of image size) allowed for retrieved ER's
    \param  minProbability    The minimum probability P(er|character) allowed for retrieved ER's
    \param  nonMaxSuppression Whenever non-maximum suppression is done over the branch probabilities
    \param  minProbability    The minimum probability difference between local maxima and local minima ERs
*/
ERFilterNM* createERFilterNM1(ERClassifierNM* erc, int thresholdDelta, float minArea, float maxArea, float minArea, float minProbability, bool nonMaxSuppression, float minProbabilityDiff)
{
	assert((minProbability >= 0.) && (minProbability <= 1.));
	assert(minArea < maxArea) && (minArea >=0.) && (maxArea <= 1.);
	assert((thresholdDelta >= 0) && (thresholdDelta <= 128));
	assert((minProbabilityDiff >= 0.) && (minProbabilityDiff <= 1.));

	ERFilterNM* filter;
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
                           default classifier can be implicitly load with function loadClassifierNM1()
                           from file in samples/cpp/trained_classifierNM2.xml
    \param  minProbability The minimum probability P(er|character) allowed for retreived ER's
*/
ERFilterNM* createERFilterNM2(ERClassifierNM* erc, float minProbability)
{
	assert(minProbability >= 0. && minProbability <= 1.)
	ERFilterNM* filter;
	setCallback(filter, erc);
	setMinProbability(filter, minProbability);
	return filter;
}


// the key method. Takes image on input, vector of ERStat is output for the first stage,
// input/output for the second one.
void run(ERFilterNM* filter, Mat m, vector* _regions)
{
	assert(type(m) == CV_8UC1);
	filter->regions = _regions;
	create(&(filter->region_mask), m.rows+2, m.cols+2, CV_8UC1);

	// if regions vector is empty we must extract the entire component tree
	if(!vector_size(filter->regions))
	{
		er_tree_extract(filter, m);
		if(filter->nonMaxSuppression)
			er_tree_nonmax_suppression(vector_front(filter->regions), NULL, NULL);
	}
	else
	{
		assert((vector_front(filter->regions))->parent == NULL);
		er_tree_filter(m, vector_front(filter->regions), NULL, NULL);
	}
}

// extract the component tree and store all the ER regions
// uses the algorithm described in
// Linear time maximally stable extremal regions, D Nistér, H Stewénius – ECCV 2008
void er_tree_extract(ERFilterNM* filter, Mat src)
{	
	assert(type(src) == CV_8UC1);

	if(filter->thresholdDelta > 1)
	{
		// ***check this***
		src.data = src.data/thresholdDelta -1;
	}
	const unsigned char * image_data = src.data;
    int width = src.cols, height = src.rows;

    // the component stack
    vector* er_stack;
    er_stack = malloc(sizeof(vector));
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
    vector* accessible_pixel_mask;
    accessible_pixel_mask = malloc(sizeof(vector));
    vector_init_n(accessible_pixel_mask, width*height);

    vector* accumulated_pixel_mask;
    accumulated_pixel_mask = malloc(sizeof(vector));
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
    ERStat* dummy;
    dummy = malloc(sizeof(ERStat));
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
            if ( !accessible_pixel_mask->items[neighbour_pixel] && (neighbour_pixel != current_pixel) )
            {

                int neighbour_level = image_data[neighbour_pixel];
                vector_set(accessible_pixel_mask, neighbour_pixel, &val);

                // if neighbour level is not lower than current level add neighbour to the boundary heap
                if (neighbour_level >= current_level)
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

        for (int edge = 0; edge < 8; edge++)
        {
            int neighbour4 = -1;
            int neighbour8 = -1;
            int cell = 0;
            switch (edge)
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
            if ((neighbour4 != -1)&&(*(bool *)vector_get(accumulated_pixel_mask, neighbour4))&&(image_data[neighbour4]<=image_data[current_pixel]))
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

        for (int p=0; p<3; p++)
        {
            for (int q=0; q<4; q++)
            {
                if ( (quad_before[0] == quads[p][q]) && ((p<2)||(q<2)) )
                    C_before[p]++;
                if ( (quad_before[1] == quads[p][q]) && ((p<2)||(q<2)) )
                    C_before[p]++;
                if ( (quad_before[2] == quads[p][q]) && ((p<2)||(q<2)) )
                    C_before[p]++;
                if ( (quad_before[3] == quads[p][q]) && ((p<2)||(q<2)) )
                    C_before[p]++;

                if ( (quad_after[0] == quads[p][q]) && ((p<2)||(q<2)) )
                    C_after[p]++;
                if ( (quad_after[1] == quads[p][q]) && ((p<2)||(q<2)) )
                    C_after[p]++;
                if ( (quad_after[2] == quads[p][q]) && ((p<2)||(q<2)) )
                    C_after[p]++;
                if ( (quad_after[3] == quads[p][q]) && ((p<2)||(q<2)) )
                    C_after[p]++;
            }
        }

        int d_C1 = C_after[0]-C_before[0];
        int d_C2 = C_after[1]-C_before[1];
        int d_C3 = C_after[2]-C_before[2];

        er_add_pixel(filter, *(ERStat **)vector_back(er_stack), x, y, non_boundary_neighbours, non_boundary_neighbours_horiz, d_C1, d_C2, d_C3);
        vector_set(&accumulated_pixel_mask, current_pixel, &val);

        // if we have processed all the possible threshold levels (the hea is empty) we are done!
        if (threshold_level == (255/thresholdDelta)+1)
        {
        	// save the extracted regions into the output vector
        	filter->regions = realloc(filter->regions, filter->num_accepted_regions+1);
            er_save(filter, *(ERStat **)vector_back(er_stack), NULL, NULL);

            // clean memory
            for(int r = 0; r < vector_size(er_stack);r++)
            {
            	ERStat* stat = *(ERStat **)vector_get(er_stack, i);
            	if(stat->crossings)
            		vector_free(stat->crossings);

            	deleteERStatTree(stat);
            }
            vector_free(er_stack);

            return;
        }


        // pop the heap of boundary pixels
        current_pixel = *(int *)vector_back(&boundary_pixes[threshold_level]);
        vector_delete(&boundary_pixes[threshold_level], boundary_pixes[threshold_level].size-1);
        current_edge = *(int *)vector_back(&boundary_edges[threshold_level]);
        vector_delete(&boundary_edges[threshold_level], boundary_edges[threshold_level].size-1);

        for (; threshold_level < (255/thresholdDelta)+1; threshold_level++)
            if (!vector_empty(&boundary_pixes[threshold_level]))
                break;

        int new_level = image_data[current_pixel];

        // if the new pixel has higher grey value than the current one
        if(new_level != current_level)
        {

        	current_level = new_level;

        	// process components on the top of the stack until we reach the higher grey-level
        	while (*(ERStat **)vector_back(er_stack)->level < new_level)
        	{
        		ERStat* er = *(ERStat **)vector_back(er_stack);
        		vector_delete(er_stack, er_stack->size-1);

        		if(new_level < (*(ERStat **)vector_back(er_stack))->level)
        		{
        			ERStat* temp;
        			init_ERStat(temp, new_level, current_pixel, current_pixel%width, current_pixel/width);
        			er_merge(*(ERStat **)vector_back(er_stack), er);
        			break;
        		}

        		er_merge(*(ERStat **)vector_back(er_stack), er);
        	}

        }
    }
}

// accumulate a pixel into an ER
void er_add_pixel(ERFilterNM* filter, ERStat* parent, int x, int y, int non_border_neighbours,
                                                            int non_border_neighbours_horiz,
                                                            int d_C1, int d_C2, int d_C3)
{
	parent->area++;
    parent->perimeter += 4 - 2*non_border_neighbours;
    if(vector_crossing(parent->crossings > 0))
    {
    	if(y < parent->(rect.y))
    		vector_addfront(parent->crossings, 2);
    	else if(y > (parent->rect).y + rect.height - 1)
    		vector_add(parent->crossings, 2);
    	else
    		vector_set(parent->crossings, y - parent->rect.y, *(int *)vector_get(parent->crossings, y - parent->rect.y) + 2-2*non_border_neighbours_horiz);
    }
    else
    	vector_add(parent->crossings, 2);

    parent->euler += (d_C1 - d_C2 + 2*d_C3) / 4;

    int new_x1 = min(parent->rect.x,x);
    int new_y1 = min(parent->rect.y,y);
    int new_x1 = max(parent->rect.x + parent->rect.width -1, x);
    int new_y2 = max(parent->rect.y + parent->rect.height -1, y);
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
	parent->area += child->area;

	parent->perimeter += child->perimeter;

	for (int i=parent->rect.y; i<=min(parent->rect.y + parent->rect.height-1,child->rect.y + child->rect.height-1); i++)
        if (i-child->rect.y >= 0)
        	vector_set(parent->crossings, i-parent->rect.y, *(int *)vector_get(parent->crossings, i-parent->rect.y) + *(int *)vector_get(child->crossings, i-child->rect.y));

    for (int i=parent->rect.y-1; i>=child->rect.y; i--)
        if (i-child->rect.y < vector_size(child->crossings));
            vector_add(parent->crossings, *(int *)vector_get(child->crossings, i-child->rect.y));
        else
        {
        	int val = 0;
            vector_addfront(parent->crossings, &val);
        }

    for(int i = parent->rect.y + parent->rect.height; i < child->rect.y; i++)
    	vector_add(parent->crossings, 0);

    for(int i = max(parent->rect.y + parent->rect.height,child->rect.y); i <= child->rect.y + child->rect.height-1;i++)
    	vector_add(parent_crossings, *(int *)vector_get(child->crossings, i-child->rect.y));

    parent->euler += child->euler;

    int new_x1 = min(parent->rect.x,child->rect.x);
    int new_y1 = min(parent->rect.y,child->rect.y);
    int new_x2 = max(parent->rect.x + parent->rect.width-1,child->rect.x + child->rect.width-1);
    int new_y2 = max(parent->rect.y + parent->rect.height-1,child->rect.y + child->rect.height-1);
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
    vector_add(&m_crossings, *(int *)vector_get(child->crossings, child->rect.height/6));
    vector_add(&m_crossings, *(int *)vector_get(child->crossings, 3*child->rect.height/6));
    vector_add(&m_crossings, *(int *)vector_get(child->crossings, 5*child->rect.height/6));
    sort_3ints(&m_crossings);
    child->med_crossings = (float)(*(int *)vector_get(&m_crossings, 1));

    // free unnecessary mem
    vector_free(child->crossings);

    // recover the original grey-level
    child->level = child->level*thresholdDelta;

    // before saving calculate P(child|character) and filter if possible
    if(filter->classifier != NULL)
    	child->probability = eval(filter->classifier, *child);

    if((((filter->classifier!=NULL)?(child->probability >= minProbability):true)||(nonMaxSuppression)) &&
         ((child->area >= (minArea*filter->region_mask.rows*filter->region_mask.cols)) &&
          (child->area <= (maxArea*filter->region_mask.rows*filter->region_mask.cols)) &&
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

        if (child->prev != NULL)
            child->prev->next = child->next;

        ERStat *new_child = child->child;
        if (new_child != NULL)
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

	((ERStat *)vector_back(regions))->parent = parent;
	if(prev != NULL)
		prev->next = (ERStat *)vector_back(regions);
	
	else if(parent != NULL)
		parent->child = (ERStat *)vector_back(regions);

	ERStat *old_prev = NULL;
	ERStat *this_er  = (ERStat *)vector_back(regions);

	if (this_er->parent == NULL)
    {
       this_er->probability = 0;
    }

    if (nonMaxSuppression)
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
    for(ERStat * child = er->child; child; child = child->next)
    {
    	old_prev = er_save(child, this_er, old_prev);
    }
    return this_er;
}

// recursively walk the tree and filter (remove) regions using the callback classifier
ERStat* er_tree_filter(ERFilterNM* filter, Mat src, ERStat* stat, ERStat *parent, ERStat *prev)
{
    assert(type(src) == CV_8UC1);

    //Fill the region and calculate 2nd stage features
    Rect r;
    Point p, q;
    p.x = stat->rect.x, p.y = stat->rect.y;
    q.x = stat->rect.x + stat->rect.width + 2, q.y = stat->rect.y + stat->rect.height + 2;
    createRect(r, p, q);
    Mat region = createusingRect(filter->region_mask, *r);
    region = createusingScalar(region, 0);
    int newMaskVal = 255;
    int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
    Rect rect;
    Mat _src = createusingRect(src, stat->rect);
    Point pt = init_Point(stat->pixel%src.cols - stat->rect.x, stat->pixel/src.cols - stat->rect.y);
    Scalar s1 = init_Scalar(255, 0, 0, 0);
    Scalar s2 = init_Scalar(stat->level, 0, 0, 0);
    Scalar s3 = init_Scalar(0, 0, 0, 0);

    floodFill(&_src, &region, pt, s1, &rect, s2, s3, flags);
    Rect r1 = init_Rect(1, 1, rect.width, rect.height);
    region = createusingRect(region, r1);

    vector** contours;    // vector of vectors of Point
    vector* contour_poly; // Point
    vector* hierarchy;    // Scalar
    vector_init(contour_poly);
    vector_init(hierarchy);
    vector_init(contours); //check

    Point _pt;
    _pt.x = _pt.y = 0;
    findContours(region, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, _pt);
}

// recursively walk the tree selecting only regions with local maxima probability
ERStat* er_tree_nonmax_suppression(ERFilterNM *filter, ERStat* stat, ERStat* parent, ERStat* prev)
{
    if(stat->local_maxima || stat->parent == NULL)
    {
        vector_add(filter->regions, stat);

        vector_back(filter->regions)->parent = parent;
        vector_back(filter->regions)->next = NULL;
        vector_back(filter->regions)->child = NULL;

        if(prev != NULL)
            prev->next = vector_back(filter->regions);

        else if(parent != NULL)
            parent->child = vector_back(filter->regions);

        ERStat *old_prev = NULL;
        ERStat *this_er  = vector_back(regions);

        for (ERStat * child = stat->child; child; child = child->next)
        {
            old_prev = er_tree_nonmax_suppression( child, this_er, old_prev );
        }

        return this_er;
    }
    else
    {
        filter->num_rejected_regions++;
        filter->num_accepted_regions--;

        ERStat *old_prev = prev;

        for (ERStat * child = stat->child; child; child = child->next)
        {
            old_prev = er_tree_nonmax_suppression( child, parent, old_prev );
        }

        return old_prev;
    }
}

static void deleteERStatTree(ERStat* root)
{
    vector to_delete;
    vector_add(root);
    while(!vector_empty(&to_delete))
    {
        ERStat* n = *(ERStat **)vector_front(&to_delete);
        vector_delete(&to_delete);
        ERStat* c = n->child;
        if(c != NULL)
        {
            vector_add(&c);
            ERStat* sibling = c->next;
            while(sibling != NULL)
            {
                vector_add(&sibling);
                sibling = sibling->next;
            }
        }
        free(n);
    }
}

void erGroupingNM(Mat img /* Mat */, vector src/* Mat:vector */, vector** regions /* ERStat: vector of vectors */, vector** out_groups /* Vec2i: vector of vectors */, vector* out_boxes /* vector of rects */, bool d0_feedback_loop)
{
    assert(!vector_empty(&src));
    size_t num_channels = vector_size(&src);

    //process each channel independently
    for(size_t c = 0; c < num_channels; c++)
    {
        //store indices to regions in a single vector
        vector all_regions;
        vector_init(&all_regions);
        for(size_t r = 0; r < vector_size(regions[c]); r++)
        {
            vector_add(&all_regions, &init_Point((int)c, (int)r));
        }

        vector valid_pairs;
        vector_init(&valid_pairs);

        //check this
        Mat mask = Mat::zeros(img.rows+2, img.cols+2, CV_8UC1);
        Mat grey, lab;
        cvtColor(img, lab, COLOR_RGB2Lab);
        cvtColor(img, grey, COLOR_RGB2GRAY);

        //check every possible pair of regions
        for(size_t i = 0; i < vector_size(&all_regions); i++)
        {
            vector i_siblings;
            vector_init(&i_siblings);
            int first_i_sibling_idx = vector_size(&valid_pairs);
            for(size_t j = i+1; j < vector_size(&all_regions); j++)
            {
                // check height ratio, centroid angle and region distance normalized by region width
                // fall within a given interval
                if (isValidPair(grey, lab, mask, src, regions, *(Point *)vector_get(all_regions, i), *(Point *)vector_get(all_regions, j)));
                {

                }
            }
        }
    }
}

// Evaluates if a pair of regions is valid or not
// using thresholds learned on training (defined above)
bool isValidPair(Mat grey, Mat lab, Mat mask, vector channels, vector** regions, Point idx1, Point idx2)
{
    Rect minarearect = performOR(&((*(ERStat *)vector_get(regions[idx1.col], idx1.row)).rect) | &((*(ERStat *)vector_get(regions[idx2.col], idx2.row)).rect));

    // Overlapping regions are not valid pair in any case
    if(equalRects(minarearect, ((*(ERStat *)vector_get(regions[idx1.col], idx1.row)).rect)) || equalRects(minarearect, ((*(ERStat *)vector_get(regions[idx2.col], idx2.row)).rect)))
        return false;

    ERStat *i, *j;
    if(((*(ERStat *)vector_get(regions[idx1.col], idx1.row)).rect).x < ((*(ERStat *)vector_get(regions[idx2.col], idx2.row)).rect).x )
    {
        i = (ERStat *)vector_get(regions[idx1.col], idx1.row);
        j = (ERStat *)vector_get(regions[idx2.col], idx2.row);
    }
    else
    {
        i = (ERStat *)vector_get(regions[idx2.col], idx2.row);
        j = (ERStat *)vector_get(regions[idx1.col], idx1.row);
    }

    if(j->rect.x == i->rect.x)
        return false;

    float height_ratio = (float)min(i->rect.height,j->rect.height) /
                                max(i->rect.height,j->rect.height);

    Point center_i, center_j;
    center_i = init_Point(i->rect.x+i->rect.width/2, i->rect.y+i->rect.height/2);
    center_j = init_Point(j->rect.x+j->rect.width/2, j->rect.y+j->rect.height/2);
    float centroid_angle = atan2f((float)(center_j.row-center_i.row), (float)(center_j.col-center_i.col));

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

    i = (ERStat *)vector_get(regions[idx1.col], idx1.row);
    j = (ERStat *)vector_get(regions[idx2.col], idx2.row);

    Mat region = mask(Rect(i->rect.tl(),
                           i->rect.br()+ Point(2,2)));
    region = Scalar(0);
    int newMaskVal = 255;
    int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;

}