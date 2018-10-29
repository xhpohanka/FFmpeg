/*
 * Copyright (c) 2008 Affine Systems, Inc (Michael Sullivan, Bobby Impollonia)
 * Copyright (c) 2013 Andrey Utkin <andrey.krieger.utkin gmail com>
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * Box and grid drawing filters. Also a nice template for a filter
 * that needs to write in the input frame.
 */

#include "config.h"
#if HAVE_OPENCV2_CORE_CORE_C_H
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#else
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#endif
#include "libavutil/colorspace.h"
#include "libavutil/common.h"
#include "libavutil/opt.h"
#include "libavutil/eval.h"
#include "libavutil/pixdesc.h"
#include "libavutil/parseutils.h"
#include "avfilter.h"
#include "formats.h"
#include "internal.h"
#include "video.h"

static const char *const var_names[] = {
    "dar",
    "hsub", "vsub",
    "in_h", "ih",      ///< height of the input video
    "in_w", "iw",      ///< width  of the input video
    "sar",
    "w",              ///< line width
    "t",
    "max",
    NULL
};

enum { Y, U, V, A };

enum var_name {
    VAR_DAR,
    VAR_HSUB, VAR_VSUB,
    VAR_IN_H, VAR_IH,
    VAR_IN_W, VAR_IW,
    VAR_SAR,
    VAR_W,
    VAR_T,
    VAR_MAX,
    VARS_NB
};

typedef struct DrawBBoxContext {
    const AVClass *class;
    int thickness;
    char *color_str;
    unsigned char yuv_color[4];
    int invert_color; ///< invert luma color
    int vsub, hsub;   ///< chroma subsampling
    char *thresh_expr;
    float thresh;
    char *w_expr;          ///< expression for line width
    char *filename;
    FILE *afile;
    int offset;
    int *frpos;
    int have_alpha;
} DrawBBoxContext;

static const int NUM_EXPR_EVALS = 5;

static CvFont font;

static av_cold int init(AVFilterContext *ctx)
{
    int ret;
    DrawBBoxContext *s = ctx->priv;
    uint8_t rgba_color[4];
    char line[256];
    char *lptr;
    int nof;
    float currfr;

    if (!strcmp(s->color_str, "invert"))
        s->invert_color = 1;
    else if (av_parse_color(rgba_color, s->color_str, -1, ctx) < 0)
        return AVERROR(EINVAL);

    if (!s->invert_color) {
        s->yuv_color[Y] = RGB_TO_Y_CCIR(rgba_color[0], rgba_color[1], rgba_color[2]);
        s->yuv_color[U] = RGB_TO_U_CCIR(rgba_color[0], rgba_color[1], rgba_color[2], 0);
        s->yuv_color[V] = RGB_TO_V_CCIR(rgba_color[0], rgba_color[1], rgba_color[2], 0);
        s->yuv_color[A] = rgba_color[3];
    }

    if (!s->filename) {
        av_log(ctx, AV_LOG_ERROR, "Filename must be set.\n");
        return AVERROR(EINVAL);
    }

    s->afile = fopen(s->filename, "r");
    if (!s->afile) {
        ret = AVERROR(errno);
        av_log(ctx, AV_LOG_ERROR, "%s: %s\n", s->filename, av_err2str(ret));
        return ret;
    }

    fseek(s->afile, -90, SEEK_END);
    fread(line, 1, 90, s->afile);
    line[90] = '\0';
    lptr = strchr(line, '\n');
    lptr++;
    sscanf(lptr, "%f", &currfr);
    fseek(s->afile, 0, SEEK_SET);

    if ((int) currfr <= 0) {
        av_log(ctx, AV_LOG_ERROR, "error in parsing file");
        return AVERROR(EINVAL);
    }

    nof = (int) currfr;

    s->frpos = av_malloc(sizeof(*s->frpos) * nof);
    if (s->frpos == NULL)
        return AVERROR(ENOMEM);
    for (int fr = 0; fr < nof; fr++)
        s->frpos[fr] = -1;


    currfr = 0.0f;
    {
        int lines = 0;
        int last = 0;

        while (currfr < nof) {
            int fr;
            int pos = ftell(s->afile);
            lptr = fgets(line, sizeof(line), s->afile);

            sscanf(line, "%f", &currfr);
            fr = (int) currfr - 1;
            fr += s->offset;
            if (fr >= nof) {
                printf("xxx");
                break;
            }

            if (last != fr && fr >= 0) {
                s->frpos[fr] = pos;
            }

            last = fr;
            lines++;
        }
    }

    cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5, 0, 1, 8);

    return 0;
}

static void uninit(AVFilterContext *ctx)
{
    DrawBBoxContext *s = ctx->priv;
    if (s->afile)
        fclose(s->afile);
    if (s->frpos)
        av_free(s->frpos);
}

static void fill_iplimage_from_frame(IplImage *img, const AVFrame *frame, enum AVPixelFormat pixfmt)
{
    IplImage *tmpimg;
    int depth, channels_nb;

    if      (pixfmt == AV_PIX_FMT_GRAY8) { depth = IPL_DEPTH_8U;  channels_nb = 1; }
    else if (pixfmt == AV_PIX_FMT_BGRA)  { depth = IPL_DEPTH_8U;  channels_nb = 4; }
    else if (pixfmt == AV_PIX_FMT_BGR24) { depth = IPL_DEPTH_8U;  channels_nb = 3; }
    else return;

    tmpimg = cvCreateImageHeader((CvSize){frame->width, frame->height}, depth, channels_nb);
    *img = *tmpimg;
    cvReleaseImageHeader(&tmpimg);
    img->imageData = img->imageDataOrigin = frame->data[0];
    img->dataOrder = IPL_DATA_ORDER_PIXEL;
    img->origin    = IPL_ORIGIN_TL;
    img->widthStep = frame->linesize[0];
}

static int query_formats(AVFilterContext *ctx)
{
    static const enum AVPixelFormat pix_fmts[] = {
        AV_PIX_FMT_BGR24, AV_PIX_FMT_BGRA, AV_PIX_FMT_GRAY8, AV_PIX_FMT_NONE
    };
    AVFilterFormats *fmts_list = ff_make_format_list(pix_fmts);
    if (!fmts_list)
        return AVERROR(ENOMEM);
    return ff_set_common_formats(ctx, fmts_list);
}

static int config_input(AVFilterLink *inlink)
{
    AVFilterContext *ctx = inlink->dst;
    DrawBBoxContext *s = ctx->priv;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);
    double var_values[VARS_NB], res;
    char *expr;
    int ret;
    int i;

    s->hsub = desc->log2_chroma_w;
    s->vsub = desc->log2_chroma_h;
    s->have_alpha = desc->flags & AV_PIX_FMT_FLAG_ALPHA;

    var_values[VAR_IN_H] = var_values[VAR_IH] = inlink->h;
    var_values[VAR_IN_W] = var_values[VAR_IW] = inlink->w;
    var_values[VAR_SAR]  = inlink->sample_aspect_ratio.num ? av_q2d(inlink->sample_aspect_ratio) : 1;
    var_values[VAR_DAR]  = (double)inlink->w / inlink->h * var_values[VAR_SAR];
    var_values[VAR_HSUB] = s->hsub;
    var_values[VAR_VSUB] = s->vsub;
    var_values[VAR_W] = NAN;
    var_values[VAR_T] = NAN;

    for (i = 0; i <= NUM_EXPR_EVALS; i++) {
        /* evaluate expressions, fail on last iteration */
        var_values[VAR_MAX] = INT_MAX;
        if ((ret = av_expr_parse_and_eval(&res, (expr = s->w_expr),
                                          var_names, var_values,
                                          NULL, NULL, NULL, NULL, NULL, 0, ctx)) < 0 && i == NUM_EXPR_EVALS)
            goto fail;
        s->thickness = var_values[VAR_W] = res;

        if ((ret = av_expr_parse_and_eval(&res, (expr = s->thresh_expr),
                var_names, var_values,
                NULL, NULL, NULL, NULL, NULL, 0, ctx)) < 0 && i == NUM_EXPR_EVALS)
            goto fail;
        s->thresh = var_values[VAR_T] = res;
    }

    return 0;

fail:
    av_log(ctx, AV_LOG_ERROR,
           "Error when evaluating the expression '%s'.\n",
           expr);
    return ret;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *frame)
{
    DrawBBoxContext *s = inlink->dst->priv;
    int frame_number = frame->pts * inlink->frame_rate.num / inlink->time_base.den + 1; // odhad, ale zdase, ze funguje
    char line[256];
    int lastfr;
    AVFilterLink *outlink= inlink->dst->outputs[0];
    IplImage inimg;

    if (s->frpos[frame_number - 1] < 0)
        return ff_filter_frame(inlink->dst->outputs[0], frame);

    fseek(s->afile, s->frpos[frame_number - 1], SEEK_SET);

    fill_iplimage_from_frame(&inimg , frame , inlink->format);

    lastfr = frame_number;
    while (fgets(line, sizeof(line), s->afile)) {
        char text[32];
        float currfr, prob, xmin, ymin, xmax, ymax;
        int filled = 0;
        char *lptr = line;

        sscanf(lptr, "%f %n", &currfr, &filled);
        lptr += filled;

        currfr += s->offset;
        if (currfr < 0)
            currfr = 0;

        if ((int) currfr != lastfr) {
            lastfr = currfr;
            break;
        }

        sscanf(lptr, "%f %n", &prob, &filled);
        lptr += filled;
        if (prob < s->thresh)
            continue;

        sscanf(lptr, "%f %n", &xmin, &filled);
        lptr += filled;
        sscanf(lptr, "%f %n", &ymin, &filled);
        lptr += filled;
        sscanf(lptr, "%f %n", &xmax, &filled);
        lptr += filled;
        sscanf(lptr, "%f %n", &ymax, &filled);

        //TODO: proc proboha ty body nesedi o dvojnasobek, kdyz putText funguje spravne...
        cvRectangle(&inimg, cvPoint(xmin*2, ymin*2), cvPoint(xmax*2, ymax*2), cvScalar(0, 255, 0, 255), 1, 8, 1);
        sprintf(text, "%f", prob);
        cvPutText(&inimg, text, cvPoint(xmin, ymin - 3), &font, cvScalar(0, 255, 0, 255));
    }

    return ff_filter_frame(outlink, frame);
}

#define OFFSET(x) offsetof(DrawBBoxContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM|AV_OPT_FLAG_FILTERING_PARAM

#if CONFIG_DRAWBBOX_FILTER

static const AVOption drawbbox_options[] = {
    { "threshold", "set threshold to display",                     OFFSET(thresh_expr), AV_OPT_TYPE_STRING, { .str="0.0" },   CHAR_MIN, CHAR_MAX, FLAGS},
    { "t",         "set threshold to display",                     OFFSET(thresh_expr), AV_OPT_TYPE_STRING, { .str="0.0" },   CHAR_MIN, CHAR_MAX, FLAGS},
    { "color",     "set color of the box",                         OFFSET(color_str), AV_OPT_TYPE_STRING, { .str = "black" }, CHAR_MIN, CHAR_MAX, FLAGS },
    { "c",         "set color of the box",                         OFFSET(color_str), AV_OPT_TYPE_STRING, { .str = "black" }, CHAR_MIN, CHAR_MAX, FLAGS },
    { "width",     "set the line width",                           OFFSET(w_expr),    AV_OPT_TYPE_STRING, { .str="3" },       CHAR_MIN, CHAR_MAX, FLAGS },
    { "w",         "set the line width",                           OFFSET(w_expr),    AV_OPT_TYPE_STRING, { .str="3" },       CHAR_MIN, CHAR_MAX, FLAGS },
    { "filename",  "file with bboxes",                             OFFSET(filename),  AV_OPT_TYPE_STRING, { .str=NULL },      CHAR_MIN, CHAR_MAX, FLAGS },
    { "f",         "file with bboxes",                             OFFSET(filename),  AV_OPT_TYPE_STRING, { .str=NULL },      CHAR_MIN, CHAR_MAX, FLAGS },
    { "offset",    "frame offset",                                 OFFSET(offset),    AV_OPT_TYPE_INT,    { .i64=0 },         INT_MIN, INT_MAX, FLAGS },
    { "o",         "frame offset",                                 OFFSET(offset),    AV_OPT_TYPE_INT,    { .i64=0 },         INT_MIN, INT_MAX, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(drawbbox);

static const AVFilterPad drawbbox_inputs[] = {
    {
        .name           = "default",
        .type           = AVMEDIA_TYPE_VIDEO,
        .config_props   = config_input,
        .filter_frame   = filter_frame,
        .needs_writable = 1,
    },
    { NULL }
};

static const AVFilterPad drawbbox_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
    },
    { NULL }
};

AVFilter ff_vf_drawbbox = {
    .name          = "drawbbox",
    .description   = NULL_IF_CONFIG_SMALL("Draw a colored box on the input video."),
    .priv_size     = sizeof(DrawBBoxContext),
    .priv_class    = &drawbbox_class,
    .init          = init,
    .uninit        = uninit,
    .query_formats = query_formats,
    .inputs        = drawbbox_inputs,
    .outputs       = drawbbox_outputs,
    .flags         = AVFILTER_FLAG_SUPPORT_TIMELINE_GENERIC,
};
#endif /* CONFIG_DRAWBBOX_FILTER */
