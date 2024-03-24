import os
import requests
import labelbox

def create_folder(image_path):
    CHECK_FOLDER = os.path.isdir(image_path)
    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(image_path)

def download_and_save_image(image_url, image_name, image_dir):
    image_path = os.path.join(image_dir, image_name)
    if not os.path.exists(image_path):
        response = requests.get(image_url)
        if response.status_code == 200:
            with open(image_path, 'wb') as f:
                f.write(response.content)
            return True
        else:
            print(f"Failed to download image from {image_url}")
            return False
    return True

def save_images_by_label(data, image_dir):
    label_folders = ['Restlessness', 'Irritatbily', 'Nervousness', 'Impending Doom', 'Difficulty Relaxing', 'Excessive Worry', 'Lack of Worry Control']
    for folder in label_folders:
        folder_path = os.path.join(image_dir, folder)
        create_folder(folder_path)

    for project_data in data:
        for project_id, project_info in project_data['projects'].items():
            for label_data in project_info['labels']:
                for classification in label_data['annotations']['classifications']:
                    if 'radio_answer' in classification:
                        radio_answer = classification['radio_answer']
                        label_name = radio_answer['name']
                        if label_name in label_folders:
                            image_url = project_data['data_row']['row_data']
                            image_name = f"{project_data['data_row']['external_id']}.jpg"
                            label_folder = os.path.join(image_dir, label_name)
                            if download_and_save_image(image_url, image_name, label_folder):
                                print(f"Image saved: {os.path.join(label_folder, image_name)}")

def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    image_path = os.path.join(dir_path, "images")
    create_folder(image_path)

    client = labelbox.Client(api_key='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbHQ2MmdoaGowMnlsMDd2Y2VxMHY2Ymh6Iiwib3JnYW5pemF0aW9uSWQiOiJjbHQ2MmdoaGEwMnlrMDd2Yzd1NGViaDl5IiwiYXBpS2V5SWQiOiJjbHR3OGtpbmQxMTd5MDcwcTFwMTMzamsyIiwic2VjcmV0IjoiY2RiYjlmNDA2NzFkMDljOTZlYzg0YTc5N2U3M2ExNDkiLCJpYXQiOjE3MTA3MjM0NDUsImV4cCI6MjM0MTg3NTQ0NX0.ksu2f4xC8RILe45TZoQ7nTsIKEPy8_n3S1QPNb0d47M')
    params = {
        "data_row_details": False,
        "metadata_fields": False,
        "attachments": True,
        "project_details": False,
        "performance_details": False,
        "label_details": True,
        "interpolated_frames": False
    }

    export_task = client.get_project('clt63ic3506us07wvgvx2bqfr').export_v2(params=params)
    export_task.wait_till_done()

    if export_task.errors:
        print(export_task.errors)
    else:
        export_json = export_task.result
        save_images_by_label(export_json, image_path)

if __name__ == "__main__":
    main()
