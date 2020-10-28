import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import cv2
import albumentations as albu
import math
import os

COLOR_LIST = [
    (0, 0, 0),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 255),
]


# helper function for data visualization
def visualize_img(*no_title_images, cols=1, show=True, **images):
    """PLot images in one row."""
    n = len(images) + len(no_title_images)
    rows = math.ceil(n / cols)
    plt.figure(figsize=(5 * cols, 5 * rows))
    cols = np.ceil(n / rows)
    for i, image in enumerate(no_title_images):
        plt.subplot(rows, cols, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image)

    for i, (name, image) in enumerate(images.items()):
        plt.subplot(rows, cols, len(no_title_images) + i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)

    if show:
        plt.show()


def get_heatmap(weights, featuremap, image):
    """
    绘制heatmap
    :param np.ndarray weights: 不同层的权重 C
    :param np.ndarray featuremap: featuremap C,H,W
    :param np.ndarray image: 原图 H,W,3
    :return:
    """

    heatmap = np.sum(featuremap * weights.reshape([-1, 1, 1]), axis=0)  # [B, H, W]

    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap)) * 255
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = heatmap.astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return (heatmap * 0.2 + image * 0.7).astype(np.uint8)


def visualize_reid(query, galleries, query_pid, gallery_pids):
    """可视化reid的结果
    
    Arguments:
        query {np.array} -- query image
        galleries {[np.array]} -- gallery images
        query_pid {int}} -- query的id
        gallery_pids {[int]} -- gallerys的id
    """
    transforms = albu.Compose(
        [
            # albu.SmallestMaxSize(256),
            albu.LongestMaxSize(256),
            # albu.CenterCrop(256, 256),
            albu.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_CONSTANT, value=(150, 150, 150))
            # albu.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_REPLICATE)
        ]
    )
    n = len(galleries)
    plt.figure(figsize=(4 * (n + 1), 5))
    plt.subplot(1, n + 1, 1)
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0, hspace=0)
    plt.xticks([])
    plt.yticks([])
    plt.title(query_pid)
    # query = cv2.resize(query, (256, 256))
    query = transforms(image=query)['image']
    plt.imshow(query)
    # plt.gca().add_patch(Rectangle((0, 0), query.shape[1], query.shape[0], edgecolor='w', linewidth=10, fill=False))
    for i in range(len(galleries)):
        g_img = galleries[i]
        # g_img = cv2.resize(g_img, (256,256))
        g_img = transforms(image=g_img)['image']
        g_pid = gallery_pids[i]
        plt.subplot(1, n + 1, i + 2)
        plt.xticks([])
        plt.yticks([])
        plt.title(g_pid)
        plt.imshow(g_img)
        if g_pid == query_pid:
            plt.gca().add_patch(
                Rectangle((0, 0), g_img.shape[1], g_img.shape[0], edgecolor='g', linewidth=10, fill=False))
        else:
            plt.gca().add_patch(
                Rectangle((0, 0), g_img.shape[1], g_img.shape[0], edgecolor='r', linewidth=10, fill=False))


def render_mask_to_img(img, cls_map, num_classes):
    """

    :param img:
    :param cls_map:
    :return:
    """
    img = img.copy()
    for i in range(num_classes):
        if i == 0:
            continue
        img[cls_map == i] = img[cls_map == i] * 0.7 + np.array(COLOR_LIST[i]) * 0.3

    return img


def render_keypoints_to_img(image, points, kp_vis=None, diameter=5):
    if kp_vis is not None:
        points = [point for vis, point in zip(kp_vis, points) if vis]
    im = image.copy()

    for (x, y) in points:
        cv2.circle(im, (int(x), int(y)), diameter, (0, 255, 0), -1)

    return im

def render_bboxes_to_img(image, bboxes, color=(255, 0, 0), thickness=5):
    """将bbox画到图片上
    
    Arguments:
        image {[type]} -- [description]
        bboxes {[type]} -- bbox的列表。bbox格式为左上角xy和右下角xy
    
    Keyword Arguments:
        color {tuple} -- [description] (default: {(255, 0, 0)})
        thickness {int} -- [description] (default: {10})
    """
    im = image.copy()
    for bbox in bboxes:
        pt1 = (int(bbox[0]), int(bbox[1]))
        pt2 = (int(bbox[2]), int(bbox[3]))
        cv2.rectangle(im, pt1, pt2, color, thickness)
    return im


def generate_html_table(content_table, image_width='auto', image_height='auto', output_path=''):
    """Generate html table

    Args:
        content_table: 2D table
        width: image width
        height: image height
        output_path: output html path.
    """
    html = '<html>'
    html += '<head>'

    html +="""
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
        <link href="https://cdn.bootcdn.net/ajax/libs/font-awesome/5.15.1/css/all.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://unpkg.com/bootstrap-table@1.18.0/dist/bootstrap-table.min.css">
    """

    html += '</head>'
    html += '<body>'
    html += """
        <table
            class="table"
            id="table"
            data-search="true"
            data-pagination="true"
            data-show-toggle="true"
            data-page-list="[10, 25, 50, 100, all]"
            data-show-refresh="true"
            data-show-fullscreen="true"
            data-show-columns="true"
            data-show-columns-toggle-all="true"
            data-show-export="true"
            data-click-to-select="true"
            data-minimum-count-columns="2"
            data-show-pagination-switch="true"
            data-id-field="id"
            data-show-footer="true"
        >
    """

    html += '<thead>'
    html += '<tr>'
    heads = content_table[0].keys()

    for i, h in enumerate(heads):
        html += f'<th data-field="{h}" data-sortable=true>{h}</th>'

    html += "</tr>"
    html += "</thead>"
    html += '</table>'

    html +="""
        <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.5.1/jquery.min.js" integrity="sha512-bLT0Qm9VnAYZDflyKcBaQ2gg0hSYNQrJ8RilYldYQ1FxQYoCLtUjuuRuZo+fjqhx/qtq/1itJ0C2ejDxltZVFg==" crossorigin="anonymous"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js" integrity="sha384-UO2eT0CpHqdSJQ6hJty5KVphtPhzWj9WO1clHTMGa3JDZwrnQq4sF86dIHNDz0W1" crossorigin="anonymous"></script>
        <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js" integrity="sha384-JjSmVgyd0p3pXB1rRibZUAYoIIy6OrQ6VrjIEaFf/nJGzIxFDsf4x0xIM+B07jRM" crossorigin="anonymous"></script>
        <script src="https://unpkg.com/bootstrap-table@1.18.0/dist/bootstrap-table.min.js"></script>
    """

    width = image_width
    height = image_height
    all_content_dict = []
    for content_row in content_table:
        content_dict = {}
        for i, head in enumerate(heads):
            content = content_row[head]
            subhtml = ''

            if type(content) == dict: # 图片，支持更丰富的样式
                src = content['src']
                alt = '' if not "alt" in content else content['alt']
                title = '' if not "title" in content else content['title']
                item_width = width if not "width" in content else content['width']
                item_height = height if not "height" in content else content['height']
                text = '' if not "text" in content else content['text']
                if text != '':
                    subhtml += f"<div>{text}</div>"
                subhtml += f"<img src={src} alt=\"{alt}\" title=\"{title}\" height={item_height} width={item_width}>"

            # 图片
            if type(content) == str and os.path.splitext(content)[-1].lower() in ['.jpg', '.png', '.jpeg', '.gif']:
                src = content
                subhtml += f"<img src={src} alt=\"{src}\" height={height} width={width}>"

            # 视频
            elif type(content) == str and os.path.splitext(content)[-1].lower() in ['.mp4', '.webm']:
                src = content
                subhtml += f"<video src={src} alt=\"{src}\" height={height} width={width}>"

            else:
                subhtml = f"{content}"

            content_dict[head] = subhtml
        all_content_dict.append(content_dict)

    html += """
    <script>
    var $table = $('#table')
    $(function() {
        var data = %s;     
        $table.bootstrapTable({data: data, paginationUseIntermediate: true})
    })
    </script>
    """ % str(all_content_dict)
    html += '</body></html>'
    if output_path != '':
        open(output_path, 'w').write(html)

    return html