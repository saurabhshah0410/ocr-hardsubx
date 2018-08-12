#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/*Text Recognition (OCR)*/

typedef struct OCRTesseract
{
	TessBaseAPI tess;
};

OCRTesseract* createOCRTesseract(const char* datapath, const char* language, const char* char_whitelist, int oemode, int psmode)
{
	OCRTesseract* ocr;
	const char *lang = "eng";
	if(language != NULL)
        lang = language;

    TessPageSegMode pagesegmode = (TessPageSegMode)psmode;
    TessBaseAPISetPageSegMode(&(ocr->tess), pagesegmode);

    if(char_whitelist != NULL)
		TessBaseAPISetVariable(&(ocr->tess), "tessedit_char_whitelist", char_whitelist);
    else
        TessBaseAPISetVariable(&(ocr->tess), "tessedit_char_whitelist", "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ");

    TessBaseAPISetVariable(&(ocr->tess), "save_best_choices", "T");
    return ocr;
}


/** @brief Recognize text using the tesseract-ocr API.

    Takes image on input and returns recognized text in the output_text parameter. Optionally
    provides also the Rects for individual text elements found (e.g. words), and the list of those
    text elements with their confidence values.

    @param image Input image CV_8UC1 or CV_8UC3
    @param output_text Output text of the tesseract-ocr.
    @param component_rects If provided the method will output a list of Rects for the individual
    text elements found (e.g. words or text lines).
    @param component_texts If provided the method will output a list of text strings for the
    recognition of individual text elements found (e.g. words or text lines).
    @param component_confidences If provided the method will output a list of confidence values
    for the recognition of individual text elements found (e.g. words or text lines).
    @param component_level OCR_LEVEL_WORD (by default), or OCR_LEVEL_TEXTLINE.
     */
void run_ocr(TessBaseAPI* tess, Mat image, char* output_text, int component_level/*0*/)
{
	TessBaseAPISetImage(tess, (unsigned char*)image.data, image.cols, image.rows, channels(image), (image.step[0])/CV_ELEM_SIZE1(image.flags));
	TessBaseAPIRecognize(tess, 0);
	
	output_text = TessBaseAPIGetUTF8Text(tess);
}