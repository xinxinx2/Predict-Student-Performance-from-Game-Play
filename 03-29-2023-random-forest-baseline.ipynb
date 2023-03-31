{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "34e17234",
   "metadata": {
    "_cell_guid": "2a9c184c-dda1-43f0-9100-5fcdeaf0e8f6",
    "_uuid": "e294f121-2756-4a6f-8f52-5762317096c9",
    "papermill": {
     "duration": 0.009833,
     "end_time": "2023-03-30T07:50:25.306234",
     "exception": false,
     "start_time": "2023-03-30T07:50:25.296401",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Random Forest Baseline - LB 0.664\n",
    "In this notebook we present a Random Forest baseline. We train GroupKFold models for each of the 18 questions. Our CV score is 0.664. We infer test using one of our KFold models. We can improve our CV and LB by engineering more features for our random forest and/or trying different models (like other ML models and/or RNN and/or Transformer). Also we can improve our LB by using more KFold models OR training one model using all data (and the hyperparameters that we found from our KFold cross validation)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "8c23e81a",
   "metadata": {
    "_cell_guid": "6a6bf827-c964-478e-a58e-10a77547ac24",
    "_uuid": "1be4af52-a1d5-44e8-856d-f7e83fc73dc5",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-03-30T07:50:25.325731Z",
     "iopub.status.busy": "2023-03-30T07:50:25.324955Z",
     "iopub.status.idle": "2023-03-30T07:50:26.686819Z",
     "shell.execute_reply": "2023-03-30T07:50:26.685799Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 1.374727,
     "end_time": "2023-03-30T07:50:26.689580",
     "exception": false,
     "start_time": "2023-03-30T07:50:25.314853",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "import pandas as pd, numpy as np\n",
    "from sklearn.model_selection import KFold, GroupKFold\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import f1_score\n",
    "import gc\n",
    "import random\n",
    "from tqdm import tqdm"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f24175b2",
   "metadata": {
    "_cell_guid": "09d1455e-5b36-4ef4-9a36-7e71edb9b03a",
    "_uuid": "bf6b0c16-62d0-4d50-8bd9-93a98425dcd4",
    "papermill": {
     "duration": 0.008582,
     "end_time": "2023-03-30T07:50:26.706563",
     "exception": false,
     "start_time": "2023-03-30T07:50:26.697981",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Load Train Data and Labels"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "a540df12",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-03-30T07:50:26.725228Z",
     "iopub.status.busy": "2023-03-30T07:50:26.724490Z",
     "iopub.status.idle": "2023-03-30T07:50:26.901388Z",
     "shell.execute_reply": "2023-03-30T07:50:26.899787Z"
    },
    "papermill": {
     "duration": 0.189346,
     "end_time": "2023-03-30T07:50:26.904200",
     "exception": false,
     "start_time": "2023-03-30T07:50:26.714854",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "424116\n"
     ]
    }
   ],
   "source": [
    "f_label=open('/kaggle/input/predict-student-performance-from-game-play/train_labels.csv')\n",
    "for i,line in enumerate(f_label):\n",
    "    pass\n",
    "print(i)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "e446c6b4",
   "metadata": {
    "_cell_guid": "6a4f73dc-8759-4714-a68c-fb7577e02a51",
    "_uuid": "0dcde6f2-f76a-43c3-bcf4-86cafa53a529",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-03-30T07:50:26.923463Z",
     "iopub.status.busy": "2023-03-30T07:50:26.923037Z",
     "iopub.status.idle": "2023-03-30T07:51:32.315946Z",
     "shell.execute_reply": "2023-03-30T07:51:32.315029Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 65.405472,
     "end_time": "2023-03-30T07:51:32.318370",
     "exception": false,
     "start_time": "2023-03-30T07:50:26.912898",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "session_id,index,elapsed_time,event_name,name,level,page,room_coor_x,room_coor_y,screen_coor_x,screen_coor_y,hover_duration,text,fqid,room_fqid,text_fqid,fullscreen,hq,music,level_group\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|█████████▉| 26296946/26296947 [01:05<00:00, 403032.23it/s]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 采样一部分行\n",
    "def sample_row():\n",
    "    with open('/kaggle/input/predict-student-performance-from-game-play/train.csv') as f:\n",
    "#         f_label=open('/kaggle/input/predict-student-performance-from-game-play/train_labels.csv')\n",
    "        L=26296947\n",
    "        head=f.readline()\n",
    "#         head_label=f_label.readline()\n",
    "        print(head.strip())\n",
    "        with open('/kaggle/working/train_sample_rows.csv','w') as fw:\n",
    "#             label_fw=open('/kaggle/working/train_label_sample_rows.csv','w')\n",
    "            fw.write(head)\n",
    "#             label_fw.write(head_label)\n",
    "            for line in tqdm(f,total=L):\n",
    "#                 line_label=f_label.readline()\n",
    "                if random.random()<1/5000:\n",
    "                    fw.write(line)\n",
    "#                     label_fw.write(line_label)\n",
    "#         f_label.close()\n",
    "#         label_fw.close()\n",
    "sample_row()\n",
    "gc.collect()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "c55cfb1c",
   "metadata": {
    "_cell_guid": "324b3fbe-302d-482b-9fcb-2d332f8681df",
    "_uuid": "f2486043-4565-48ed-a0e1-0974cd254e4b",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-03-30T07:51:32.416032Z",
     "iopub.status.busy": "2023-03-30T07:51:32.415249Z",
     "iopub.status.idle": "2023-03-30T07:52:57.609512Z",
     "shell.execute_reply": "2023-03-30T07:52:57.608159Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 85.245665,
     "end_time": "2023-03-30T07:52:57.612139",
     "exception": false,
     "start_time": "2023-03-30T07:51:32.366474",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[15, 14, 12]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 26296946/26296946 [01:25<00:00, 309141.83it/s]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 采样一部分列\n",
    "# 26296947\n",
    "def sample_column():\n",
    "    with open('/kaggle/input/predict-student-performance-from-game-play/train.csv') as f:\n",
    "        head=f.readline()\n",
    "    #     print(head)\n",
    "        head=head.strip().split(',')\n",
    "        remove_i=[head.index('text'),head.index('room_fqid'),head.index('text_fqid')]\n",
    "    #     remove_i=remove_i[::-1]\n",
    "        remove_i.sort(reverse=True)\n",
    "        print(remove_i)\n",
    "        for i in remove_i:\n",
    "            head.pop(i)\n",
    "        head=','.join(head)+'\\n'\n",
    "    #     print(i,head.strip().split(',')[i])\n",
    "        with open('/kaggle/working/train_sample_columns.csv','w') as fw:\n",
    "            fw.write(head)\n",
    "            for line in tqdm(f,total=26296946):\n",
    "                line=line.strip().split(',')\n",
    "                for i in remove_i:\n",
    "                    line.pop(i)\n",
    "                line=','.join(line)+'\\n'\n",
    "                fw.write(line)\n",
    "sample_column()\n",
    "gc.collect()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "c7be49af",
   "metadata": {
    "_cell_guid": "cd65ab7d-f176-4b84-a887-9e6b6c7ea184",
    "_uuid": "35367e01-0e6e-4a11-929f-b28e82e9d796",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-03-30T07:52:57.811552Z",
     "iopub.status.busy": "2023-03-30T07:52:57.810465Z",
     "iopub.status.idle": "2023-03-30T07:52:57.906957Z",
     "shell.execute_reply": "2023-03-30T07:52:57.905668Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.197481,
     "end_time": "2023-03-30T07:52:57.909987",
     "exception": false,
     "start_time": "2023-03-30T07:52:57.712506",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(5259, 20)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>session_id</th>\n",
       "      <th>index</th>\n",
       "      <th>elapsed_time</th>\n",
       "      <th>event_name</th>\n",
       "      <th>name</th>\n",
       "      <th>level</th>\n",
       "      <th>page</th>\n",
       "      <th>room_coor_x</th>\n",
       "      <th>room_coor_y</th>\n",
       "      <th>screen_coor_x</th>\n",
       "      <th>screen_coor_y</th>\n",
       "      <th>hover_duration</th>\n",
       "      <th>text</th>\n",
       "      <th>fqid</th>\n",
       "      <th>room_fqid</th>\n",
       "      <th>text_fqid</th>\n",
       "      <th>fullscreen</th>\n",
       "      <th>hq</th>\n",
       "      <th>music</th>\n",
       "      <th>level_group</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>20090314221187252</td>\n",
       "      <td>163</td>\n",
       "      <td>116451</td>\n",
       "      <td>navigate_click</td>\n",
       "      <td>undefined</td>\n",
       "      <td>3</td>\n",
       "      <td>NaN</td>\n",
       "      <td>159.503027</td>\n",
       "      <td>-140.000000</td>\n",
       "      <td>826.0</td>\n",
       "      <td>470.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>tunic.kohlcenter.halloffame</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0-4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>20090314221187252</td>\n",
       "      <td>729</td>\n",
       "      <td>872197</td>\n",
       "      <td>cutscene_click</td>\n",
       "      <td>basic</td>\n",
       "      <td>17</td>\n",
       "      <td>NaN</td>\n",
       "      <td>416.056246</td>\n",
       "      <td>-427.327551</td>\n",
       "      <td>817.0</td>\n",
       "      <td>624.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>undefined</td>\n",
       "      <td>savedteddy</td>\n",
       "      <td>tunic.historicalsociety.basement</td>\n",
       "      <td>tunic.historicalsociety.basement.savedteddy</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>13-22</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>20090314363702160</td>\n",
       "      <td>273</td>\n",
       "      <td>358499</td>\n",
       "      <td>person_click</td>\n",
       "      <td>basic</td>\n",
       "      <td>7</td>\n",
       "      <td>NaN</td>\n",
       "      <td>217.882601</td>\n",
       "      <td>-136.000000</td>\n",
       "      <td>684.0</td>\n",
       "      <td>466.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Sorry, I'm in a hurry.</td>\n",
       "      <td>worker</td>\n",
       "      <td>tunic.humanecology.frontdesk</td>\n",
       "      <td>tunic.humanecology.frontdesk.worker.intro</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>5-12</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>20090315085850788</td>\n",
       "      <td>852</td>\n",
       "      <td>2266002</td>\n",
       "      <td>navigate_click</td>\n",
       "      <td>undefined</td>\n",
       "      <td>19</td>\n",
       "      <td>NaN</td>\n",
       "      <td>-620.586508</td>\n",
       "      <td>-562.771396</td>\n",
       "      <td>581.0</td>\n",
       "      <td>758.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>tunic.wildlife.center</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>13-22</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>20090317111400710</td>\n",
       "      <td>72</td>\n",
       "      <td>99049</td>\n",
       "      <td>navigate_click</td>\n",
       "      <td>undefined</td>\n",
       "      <td>0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>-406.189819</td>\n",
       "      <td>-51.390008</td>\n",
       "      <td>73.0</td>\n",
       "      <td>361.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>tunic.historicalsociety.closet</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0-4</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          session_id  index  elapsed_time      event_name       name  level  \\\n",
       "0  20090314221187252    163        116451  navigate_click  undefined      3   \n",
       "1  20090314221187252    729        872197  cutscene_click      basic     17   \n",
       "2  20090314363702160    273        358499    person_click      basic      7   \n",
       "3  20090315085850788    852       2266002  navigate_click  undefined     19   \n",
       "4  20090317111400710     72         99049  navigate_click  undefined      0   \n",
       "\n",
       "   page  room_coor_x  room_coor_y  screen_coor_x  screen_coor_y  \\\n",
       "0   NaN   159.503027  -140.000000          826.0          470.0   \n",
       "1   NaN   416.056246  -427.327551          817.0          624.0   \n",
       "2   NaN   217.882601  -136.000000          684.0          466.0   \n",
       "3   NaN  -620.586508  -562.771396          581.0          758.0   \n",
       "4   NaN  -406.189819   -51.390008           73.0          361.0   \n",
       "\n",
       "   hover_duration                    text        fqid  \\\n",
       "0             NaN                     NaN         NaN   \n",
       "1             NaN               undefined  savedteddy   \n",
       "2             NaN  Sorry, I'm in a hurry.      worker   \n",
       "3             NaN                     NaN         NaN   \n",
       "4             NaN                     NaN         NaN   \n",
       "\n",
       "                          room_fqid  \\\n",
       "0       tunic.kohlcenter.halloffame   \n",
       "1  tunic.historicalsociety.basement   \n",
       "2      tunic.humanecology.frontdesk   \n",
       "3             tunic.wildlife.center   \n",
       "4    tunic.historicalsociety.closet   \n",
       "\n",
       "                                     text_fqid  fullscreen  hq  music  \\\n",
       "0                                          NaN           1   1      1   \n",
       "1  tunic.historicalsociety.basement.savedteddy           1   1      1   \n",
       "2    tunic.humanecology.frontdesk.worker.intro           0   0      1   \n",
       "3                                          NaN           1   0      1   \n",
       "4                                          NaN           0   0      1   \n",
       "\n",
       "  level_group  \n",
       "0         0-4  \n",
       "1       13-22  \n",
       "2        5-12  \n",
       "3       13-22  \n",
       "4         0-4  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train = pd.read_csv('/kaggle/input/train-sample-rows/train_sample_rows.csv',memory_map=True)\n",
    "print( train.shape )\n",
    "train.head()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "c5003563",
   "metadata": {
    "_cell_guid": "6bd5d1f2-de3c-484b-9eb1-ae287b7cdd44",
    "_uuid": "1adbcc3d-0f93-4367-9e5a-9e2d1a9c6ca1",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-03-30T07:52:58.110410Z",
     "iopub.status.busy": "2023-03-30T07:52:58.109663Z",
     "iopub.status.idle": "2023-03-30T07:52:59.440326Z",
     "shell.execute_reply": "2023-03-30T07:52:59.439284Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 1.435496,
     "end_time": "2023-03-30T07:52:59.443308",
     "exception": false,
     "start_time": "2023-03-30T07:52:58.007812",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(424116, 4)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>session_id</th>\n",
       "      <th>correct</th>\n",
       "      <th>session</th>\n",
       "      <th>q</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>20090312431273200_q1</td>\n",
       "      <td>1</td>\n",
       "      <td>20090312431273200</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>20090312433251036_q1</td>\n",
       "      <td>0</td>\n",
       "      <td>20090312433251036</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>20090312455206810_q1</td>\n",
       "      <td>1</td>\n",
       "      <td>20090312455206810</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>20090313091715820_q1</td>\n",
       "      <td>0</td>\n",
       "      <td>20090313091715820</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>20090313571836404_q1</td>\n",
       "      <td>1</td>\n",
       "      <td>20090313571836404</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             session_id  correct            session  q\n",
       "0  20090312431273200_q1        1  20090312431273200  1\n",
       "1  20090312433251036_q1        0  20090312433251036  1\n",
       "2  20090312455206810_q1        1  20090312455206810  1\n",
       "3  20090313091715820_q1        0  20090313091715820  1\n",
       "4  20090313571836404_q1        1  20090313571836404  1"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "targets = pd.read_csv('/kaggle/input/predict-student-performance-from-game-play/train_labels.csv')\n",
    "targets['session'] = targets.session_id.apply(lambda x: int(x.split('_')[0]) )\n",
    "targets['q'] = targets.session_id.apply(lambda x: int(x.split('_')[-1][1:]) )\n",
    "print( targets.shape )\n",
    "targets.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "eb5e1e26",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-03-30T07:52:59.640724Z",
     "iopub.status.busy": "2023-03-30T07:52:59.639973Z",
     "iopub.status.idle": "2023-03-30T07:52:59.755144Z",
     "shell.execute_reply": "2023-03-30T07:52:59.754286Z"
    },
    "papermill": {
     "duration": 0.217129,
     "end_time": "2023-03-30T07:52:59.757480",
     "exception": false,
     "start_time": "2023-03-30T07:52:59.540351",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "21"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "gc.collect()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "daa0a4af",
   "metadata": {
    "_cell_guid": "e3bc00a2-4c2e-45aa-a05e-c63d1e1abbf8",
    "_uuid": "014cbd03-7f27-4335-92ea-77e6fa05ddc8",
    "papermill": {
     "duration": 0.097582,
     "end_time": "2023-03-30T07:52:59.952945",
     "exception": false,
     "start_time": "2023-03-30T07:52:59.855363",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Feature Engineer\n",
    "We create basic aggregate features. Try creating more features to boost CV and LB!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "faf8f842",
   "metadata": {
    "_cell_guid": "68ba1532-5617-492a-902e-5da988ccbe94",
    "_uuid": "ed3f8131-eff9-463f-86fd-7c94a49a9343",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-03-30T07:53:00.148395Z",
     "iopub.status.busy": "2023-03-30T07:53:00.147977Z",
     "iopub.status.idle": "2023-03-30T07:53:00.152668Z",
     "shell.execute_reply": "2023-03-30T07:53:00.151761Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.104662,
     "end_time": "2023-03-30T07:53:00.154779",
     "exception": false,
     "start_time": "2023-03-30T07:53:00.050117",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "CATS = ['event_name', 'name','fqid', 'room_fqid', 'text_fqid']\n",
    "NUMS = ['elapsed_time','level','page','room_coor_x', 'room_coor_y', \n",
    "        'screen_coor_x', 'screen_coor_y', 'hover_duration']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "f578c733",
   "metadata": {
    "_cell_guid": "230a4a48-e215-418f-a1b1-ac324fa5a540",
    "_uuid": "07947b86-31a7-4f99-90bc-3750334cecd5",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-03-30T07:53:00.350106Z",
     "iopub.status.busy": "2023-03-30T07:53:00.348942Z",
     "iopub.status.idle": "2023-03-30T07:53:00.357213Z",
     "shell.execute_reply": "2023-03-30T07:53:00.356266Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.108344,
     "end_time": "2023-03-30T07:53:00.360139",
     "exception": false,
     "start_time": "2023-03-30T07:53:00.251795",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def feature_engineer(train):\n",
    "    dfs = []\n",
    "    for c in CATS:\n",
    "        tmp = train.groupby(['session_id','level_group'])[c].agg('nunique')\n",
    "        tmp.name = tmp.name + '_nunique'\n",
    "        dfs.append(tmp)\n",
    "    for c in NUMS:\n",
    "        tmp = train.groupby(['session_id','level_group'])[c].agg('mean')\n",
    "        dfs.append(tmp)\n",
    "    for c in NUMS:\n",
    "        tmp = train.groupby(['session_id','level_group'])[c].agg('std')\n",
    "        tmp.name = tmp.name + '_std'\n",
    "        dfs.append(tmp)\n",
    "    df = pd.concat(dfs,axis=1)\n",
    "    df = df.fillna(-1)\n",
    "    df = df.reset_index()\n",
    "    df = df.set_index('session_id')\n",
    "    return df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "3c68ced7",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-03-30T07:53:00.627876Z",
     "iopub.status.busy": "2023-03-30T07:53:00.626852Z",
     "iopub.status.idle": "2023-03-30T07:53:00.632897Z",
     "shell.execute_reply": "2023-03-30T07:53:00.632146Z"
    },
    "papermill": {
     "duration": 0.107812,
     "end_time": "2023-03-30T07:53:00.635045",
     "exception": false,
     "start_time": "2023-03-30T07:53:00.527233",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(5259, 20)"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "c4df896d",
   "metadata": {
    "_cell_guid": "1eff2ee2-923c-400a-b86c-e2a7b688c1dc",
    "_uuid": "d84f4a3a-8c3a-4256-a111-1d05319f742c",
    "execution": {
     "iopub.execute_input": "2023-03-30T07:53:00.831900Z",
     "iopub.status.busy": "2023-03-30T07:53:00.831156Z",
     "iopub.status.idle": "2023-03-30T07:53:00.975921Z",
     "shell.execute_reply": "2023-03-30T07:53:00.974908Z"
    },
    "papermill": {
     "duration": 0.245146,
     "end_time": "2023-03-30T07:53:00.978009",
     "exception": false,
     "start_time": "2023-03-30T07:53:00.732863",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(5009, 22)\n",
      "CPU times: user 94.9 ms, sys: 5.86 ms, total: 101 ms\n",
      "Wall time: 115 ms\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>level_group</th>\n",
       "      <th>event_name_nunique</th>\n",
       "      <th>name_nunique</th>\n",
       "      <th>fqid_nunique</th>\n",
       "      <th>room_fqid_nunique</th>\n",
       "      <th>text_fqid_nunique</th>\n",
       "      <th>elapsed_time</th>\n",
       "      <th>level</th>\n",
       "      <th>page</th>\n",
       "      <th>room_coor_x</th>\n",
       "      <th>...</th>\n",
       "      <th>screen_coor_y</th>\n",
       "      <th>hover_duration</th>\n",
       "      <th>elapsed_time_std</th>\n",
       "      <th>level_std</th>\n",
       "      <th>page_std</th>\n",
       "      <th>room_coor_x_std</th>\n",
       "      <th>room_coor_y_std</th>\n",
       "      <th>screen_coor_x_std</th>\n",
       "      <th>screen_coor_y_std</th>\n",
       "      <th>hover_duration_std</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>session_id</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>20090314221187252</th>\n",
       "      <td>0-4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>116451.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>159.503027</td>\n",
       "      <td>...</td>\n",
       "      <td>470.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20090314221187252</th>\n",
       "      <td>13-22</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>872197.0</td>\n",
       "      <td>17.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>416.056246</td>\n",
       "      <td>...</td>\n",
       "      <td>624.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20090314363702160</th>\n",
       "      <td>5-12</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>358499.0</td>\n",
       "      <td>7.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>217.882601</td>\n",
       "      <td>...</td>\n",
       "      <td>466.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20090315085850788</th>\n",
       "      <td>13-22</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>2266002.0</td>\n",
       "      <td>19.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-620.586508</td>\n",
       "      <td>...</td>\n",
       "      <td>758.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20090317111400710</th>\n",
       "      <td>0-4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>99049.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-406.189819</td>\n",
       "      <td>...</td>\n",
       "      <td>361.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>-1.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 22 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                  level_group  event_name_nunique  name_nunique  fqid_nunique  \\\n",
       "session_id                                                                      \n",
       "20090314221187252         0-4                   1             1             0   \n",
       "20090314221187252       13-22                   1             1             1   \n",
       "20090314363702160        5-12                   1             1             1   \n",
       "20090315085850788       13-22                   1             1             0   \n",
       "20090317111400710         0-4                   1             1             0   \n",
       "\n",
       "                   room_fqid_nunique  text_fqid_nunique  elapsed_time  level  \\\n",
       "session_id                                                                     \n",
       "20090314221187252                  1                  0      116451.0    3.0   \n",
       "20090314221187252                  1                  1      872197.0   17.0   \n",
       "20090314363702160                  1                  1      358499.0    7.0   \n",
       "20090315085850788                  1                  0     2266002.0   19.0   \n",
       "20090317111400710                  1                  0       99049.0    0.0   \n",
       "\n",
       "                   page  room_coor_x  ...  screen_coor_y  hover_duration  \\\n",
       "session_id                            ...                                  \n",
       "20090314221187252  -1.0   159.503027  ...          470.0            -1.0   \n",
       "20090314221187252  -1.0   416.056246  ...          624.0            -1.0   \n",
       "20090314363702160  -1.0   217.882601  ...          466.0            -1.0   \n",
       "20090315085850788  -1.0  -620.586508  ...          758.0            -1.0   \n",
       "20090317111400710  -1.0  -406.189819  ...          361.0            -1.0   \n",
       "\n",
       "                   elapsed_time_std  level_std  page_std  room_coor_x_std  \\\n",
       "session_id                                                                  \n",
       "20090314221187252              -1.0       -1.0      -1.0             -1.0   \n",
       "20090314221187252              -1.0       -1.0      -1.0             -1.0   \n",
       "20090314363702160              -1.0       -1.0      -1.0             -1.0   \n",
       "20090315085850788              -1.0       -1.0      -1.0             -1.0   \n",
       "20090317111400710              -1.0       -1.0      -1.0             -1.0   \n",
       "\n",
       "                   room_coor_y_std  screen_coor_x_std  screen_coor_y_std  \\\n",
       "session_id                                                                 \n",
       "20090314221187252             -1.0               -1.0               -1.0   \n",
       "20090314221187252             -1.0               -1.0               -1.0   \n",
       "20090314363702160             -1.0               -1.0               -1.0   \n",
       "20090315085850788             -1.0               -1.0               -1.0   \n",
       "20090317111400710             -1.0               -1.0               -1.0   \n",
       "\n",
       "                   hover_duration_std  \n",
       "session_id                             \n",
       "20090314221187252                -1.0  \n",
       "20090314221187252                -1.0  \n",
       "20090314363702160                -1.0  \n",
       "20090315085850788                -1.0  \n",
       "20090317111400710                -1.0  \n",
       "\n",
       "[5 rows x 22 columns]"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "%%time\n",
    "df = feature_engineer(train)\n",
    "print( df.shape )\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b81ecdfb",
   "metadata": {
    "_cell_guid": "39d90dfc-1f13-4a4a-b5fd-8dc7bd9e80d1",
    "_uuid": "ce841bae-e51f-4bc0-8d84-d6b4afe2cadc",
    "papermill": {
     "duration": 0.097253,
     "end_time": "2023-03-30T07:53:01.172550",
     "exception": false,
     "start_time": "2023-03-30T07:53:01.075297",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Train Random Forest Model\n",
    "We train one model for each of 18 questions. Furthermore, we use data from `level_groups = '0-4'` to train model for questions 1-3, and `level groups '5-12'` to train questions 4 thru 13 and `level groups '13-22'` to train questions 14 thru 18. Because this is the data we get (to predict corresponding questions) from Kaggle's inference API during test inference. We can improve our model by saving a user's previous data from earlier `level_groups` and using that to predict future `level_groups`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "a3869036",
   "metadata": {
    "_cell_guid": "228b3508-e8ed-4e01-884c-b8caac2e1b3d",
    "_uuid": "759e4f59-eb65-413b-be09-81b7b87c2bff",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-03-30T07:53:01.368992Z",
     "iopub.status.busy": "2023-03-30T07:53:01.368170Z",
     "iopub.status.idle": "2023-03-30T07:53:01.377454Z",
     "shell.execute_reply": "2023-03-30T07:53:01.376482Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.110329,
     "end_time": "2023-03-30T07:53:01.380003",
     "exception": false,
     "start_time": "2023-03-30T07:53:01.269674",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "We will train with 21 features\n",
      "We will train with 4707 users info\n"
     ]
    }
   ],
   "source": [
    "FEATURES = [c for c in df.columns if c != 'level_group']\n",
    "print('We will train with', len(FEATURES) ,'features')\n",
    "ALL_USERS = df.index.unique()\n",
    "print('We will train with', len(ALL_USERS) ,'users info')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "dc8a87c1",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-03-30T07:53:01.578336Z",
     "iopub.status.busy": "2023-03-30T07:53:01.577550Z",
     "iopub.status.idle": "2023-03-30T07:53:01.583799Z",
     "shell.execute_reply": "2023-03-30T07:53:01.582824Z"
    },
    "papermill": {
     "duration": 0.108264,
     "end_time": "2023-03-30T07:53:01.586304",
     "exception": false,
     "start_time": "2023-03-30T07:53:01.478040",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(424116, 4)"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "targets.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "28170b6f",
   "metadata": {
    "_cell_guid": "e54bf696-4dd3-46ea-9912-59c7f0f8f11d",
    "_uuid": "32e91686-2eb2-4c83-9452-4a598f7b4df5",
    "execution": {
     "iopub.execute_input": "2023-03-30T07:53:01.784135Z",
     "iopub.status.busy": "2023-03-30T07:53:01.783392Z",
     "iopub.status.idle": "2023-03-30T07:53:36.276798Z",
     "shell.execute_reply": "2023-03-30T07:53:36.275673Z"
    },
    "papermill": {
     "duration": 34.595014,
     "end_time": "2023-03-30T07:53:36.280340",
     "exception": false,
     "start_time": "2023-03-30T07:53:01.685326",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "#########################\n",
      "### Fold 1\n",
      "#########################\n",
      "1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 14 , 15 , 16 , 17 , 18 , \n",
      "#########################\n",
      "### Fold 2\n",
      "#########################\n",
      "1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 14 , 15 , 16 , 17 , 18 , \n",
      "#########################\n",
      "### Fold 3\n",
      "#########################\n",
      "1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 14 , 15 , 16 , 17 , 18 , \n",
      "#########################\n",
      "### Fold 4\n",
      "#########################\n",
      "1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 14 , 15 , 16 , 17 , 18 , \n",
      "#########################\n",
      "### Fold 5\n",
      "#########################\n",
      "1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 14 , 15 , 16 , 17 , 18 , \n"
     ]
    }
   ],
   "source": [
    "gkf = GroupKFold(n_splits=5)\n",
    "oof = pd.DataFrame(data=np.zeros((len(ALL_USERS),18)), index=ALL_USERS)\n",
    "models = {}\n",
    "\n",
    "# COMPUTE CV SCORE WITH 5 GROUP K FOLD\n",
    "for i, (train_index, test_index) in enumerate(gkf.split(X=df, groups=df.index)):\n",
    "    print('#'*25)\n",
    "    print('### Fold',i+1)\n",
    "    print('#'*25)\n",
    "    \n",
    "    # ITERATE THRU QUESTIONS 1 THRU 18\n",
    "    for t in range(1,19):\n",
    "        print(t,', ',end='')\n",
    "        \n",
    "        # USE THIS TRAIN DATA WITH THESE QUESTIONS\n",
    "        if t<=3: grp = '0-4'\n",
    "        elif t<=13: grp = '5-12'\n",
    "        elif t<=22: grp = '13-22'\n",
    "            \n",
    "        # TRAIN DATA\n",
    "        train_x = df.iloc[train_index]\n",
    "        train_x = train_x.loc[train_x.level_group == grp]\n",
    "        train_users = train_x.index.values\n",
    "        train_y = targets.loc[targets.q==t].set_index('session').loc[train_users]\n",
    "        \n",
    "        # VALID DATA\n",
    "        valid_x = df.iloc[test_index]\n",
    "        valid_x = valid_x.loc[valid_x.level_group == grp]\n",
    "        valid_users = valid_x.index.values\n",
    "        valid_y = targets.loc[targets.q==t].set_index('session').loc[valid_users]\n",
    "        \n",
    "        # TRAIN MODEL\n",
    "        clf = RandomForestClassifier() \n",
    "        clf.fit(train_x[FEATURES].astype('float32'), train_y['correct'])\n",
    "        \n",
    "        # SAVE MODEL, PREDICT VALID OOF\n",
    "        models[f'{grp}_{t}'] = clf\n",
    "        oof.loc[valid_users, t-1] = clf.predict_proba(valid_x[FEATURES].astype('float32'))[:,1]\n",
    "        \n",
    "    print()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3ca2311c",
   "metadata": {
    "_cell_guid": "c8a77695-1c65-4d65-85da-76a2af586468",
    "_uuid": "9e03f391-fa71-4820-ba58-84a2006d95fc",
    "papermill": {
     "duration": 0.103579,
     "end_time": "2023-03-30T07:53:36.488241",
     "exception": false,
     "start_time": "2023-03-30T07:53:36.384662",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Compute CV Score\n",
    "We need to convert prediction probabilities into `1s` and `0s`. The competition metric is F1 Score which is the harmonic mean of precision and recall. Let's find the optimal threshold for `p > threshold` when to predict `1` and when to predict `0` to maximize F1 Score."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "64ee4744",
   "metadata": {
    "_cell_guid": "2fe38d25-a701-4b0c-9432-b501aac8fa15",
    "_uuid": "dae790d0-b322-4de5-b7dd-fd42be63e3f7",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-03-30T07:53:36.697190Z",
     "iopub.status.busy": "2023-03-30T07:53:36.696185Z",
     "iopub.status.idle": "2023-03-30T07:53:36.792658Z",
     "shell.execute_reply": "2023-03-30T07:53:36.791420Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.203117,
     "end_time": "2023-03-30T07:53:36.795331",
     "exception": false,
     "start_time": "2023-03-30T07:53:36.592214",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# PUT TRUE LABELS INTO DATAFRAME WITH 18 COLUMNS\n",
    "true = oof.copy()\n",
    "for k in range(18):\n",
    "    # GET TRUE LABELS\n",
    "    tmp = targets.loc[targets.q == k+1].set_index('session').loc[ALL_USERS]\n",
    "    true[k] = tmp.correct.values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "4f6e3ed3",
   "metadata": {
    "_cell_guid": "e401d214-5382-4b4e-868c-5f61dd3986e7",
    "_uuid": "696f6a93-c59d-49e4-94e3-3e31f4fa5c4a",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-03-30T07:53:37.003377Z",
     "iopub.status.busy": "2023-03-30T07:53:37.002493Z",
     "iopub.status.idle": "2023-03-30T07:53:38.479989Z",
     "shell.execute_reply": "2023-03-30T07:53:38.478790Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 1.584116,
     "end_time": "2023-03-30T07:53:38.482335",
     "exception": false,
     "start_time": "2023-03-30T07:53:36.898219",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.80, "
     ]
    }
   ],
   "source": [
    "# FIND BEST THRESHOLD TO CONVERT PROBS INTO 1s AND 0s\n",
    "scores = []; thresholds = []\n",
    "best_score = 0; best_threshold = 0\n",
    "\n",
    "for threshold in np.arange(0.4,0.81,0.01):\n",
    "    print(f'{threshold:.02f}, ',end='')\n",
    "    preds = (oof.values.reshape((-1))>threshold).astype('int')\n",
    "    m = f1_score(true.values.reshape((-1)), preds, average='macro')   \n",
    "    scores.append(m)\n",
    "    thresholds.append(threshold)\n",
    "    if m>best_score:\n",
    "        best_score = m\n",
    "        best_threshold = threshold"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "9e4aa494",
   "metadata": {
    "_cell_guid": "f021b5bd-2897-4732-9f09-33974ad06e74",
    "_uuid": "ab6399c3-23ea-454c-8db9-28e9a496b1fa",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-03-30T07:53:38.691707Z",
     "iopub.status.busy": "2023-03-30T07:53:38.690701Z",
     "iopub.status.idle": "2023-03-30T07:53:38.990785Z",
     "shell.execute_reply": "2023-03-30T07:53:38.989677Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.406666,
     "end_time": "2023-03-30T07:53:38.993342",
     "exception": false,
     "start_time": "2023-03-30T07:53:38.586676",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABJkAAAFVCAYAAABM2D5DAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8qNh9FAAAACXBIWXMAAAsTAAALEwEAmpwYAABhJ0lEQVR4nO3dd5xcVfn48c+zSQhZegkIgexSAggCAgFELDQBUSmCWKIIggEULF8QBSwgRiyIBRAMSBEDSJOm0qT5oweE0KUlIYQSOiEhhOT8/jh32clkZne2zm7283697mtmbpn7zNx7Z3eeOec5kVJCkiRJkiRJ6oqGegcgSZIkSZKk/s8kkyRJkiRJkrrMJJMkSZIkSZK6zCSTJEmSJEmSuswkkyRJkiRJkrrMJJMkSZIkSZK6zCSTpF4TEdtERIqIfesdS7mejC0iboqIyfWOQ4u+iGguzp9jemJ9SerLIuLsiEj1jqOSnoqtE5/7ffY9krRoMMkkqdOKf2pqnZrrHa+6VzvH+wcl620REX+IiFsjYmZ3JNEi4n0RcUJEPBgRb0bE6xHxeERcEBGf7fKLW4RExLIRcUxEbNOD+5hcdvznR8TzEXFzRHyhp/Zbsv/m4jV+sAPb7NvOOfy+knUPjIgJEfFoRMzrji9oEfGxiLiieO/mRMQLETExIn4fEWt29fkXRRGxTEScFBHPRsTbEfFQRBwcEdHJ52uMiKeL431yDet/o+T8WLHC8iUj4qiIeKD4XHopIm4rzrVOxdhOPMdExO4dWL85Fj7P34mIKRFxcURs2d0xVohhmyLuZWtYt71rtHS6qadjV/eJiH0i4r8RMbv47DsjIoZ34fl+VZwHM7szTkmdM7jeAUjq175S9vijwFhgPPCfsmUzgOZeiEm96z7gNxXm/7fk/i7AN4FHgfuBD3dlhxHRBNwFLA1MAE4tFq0NfApYEri0K/vox6YAw4B3S+YtC/ykuH9TD+57GnBkcX8QMAL4KnB+RKySUvptD+67mfwaJ5PPyY74A3B3hfmvldw/EliBfF4vAazWwX0sICIOBv4IPAWcAzwDDAc+AOxH/vx8qiv7WNRExGLAdcAmwEnAI8Anye/jysAxnXjanwILJYuq7H9V4HhgJvkzpnx5A/Av8ufbOUWMjcAXgbOA9wPf70SMbflJsa/LOrjddcBfivtDgXXIf7t3i4itU0p3dVuEC9uGHPfZLHiNVXILC/+fcTSwXoX5L3Q9NPWGiPgucCJwM/Bt8ufp/wFbRcQWKaW3Ovh8HwS+S742uz2ZK6njTDJJ6rSU0l9LH0fEYPI/qreXLyuWd3mfEbFUSunNLj+RusuzlY51mVOBX6eU3oqIvehikgk4HFgJ2D2ldHnZsu9GRJcSAJ3VF87NlFIC3q7T7l+v8JnwJ+A5YF+gJ5NMXfGflNLF7ayzDTA1pTQ/Iq6iC0mm4nPy58BUYJOU0htly4eRE4W9KiIGAUNTSrN6e981OgDYHPhWSumkYt7pEXEJcFREnJVSmlLrk0XEpsB3gCOonCgvdwo58fcg8OUKy7cEPgL8LqX03ZL9/JGcYD+Q7k8yddb/KlyrtwKXA2PISfy6Syk9RVmyNSIOANar4e9OpxQtzpZIKdkipgcULQB/Rk7sb59SmlfMvxu4gpx0+nkHnm8QcDo5wbs0MLq7Y5bUcXaXk1QXEbFf0dVhTtFU/4gK60yOXM9ok4i4JiJeByaVLB8VEedGxHNFk//JEfHriFii7HlWj4gzi/3MiYgXiy4MX+1sbMV6u0drF7CZxf3dOvAe7FY0F387Ip6JiJ8CQ2rc9pdF0/CNKixbpmiCflnJvE9F7rr0UrFsakRcGhHr1BpvZ6WUXujoL5PtGFXc/rvK/qaVzyvOoYuKZvlzivf7/IhYq2y9AyLi3uI9ej0iro2Ij1R4vhS5rsX2EfH/iib6V5YsHx0Rfy/e7zkR8VhEHF0kGNoUFWp4RcSXin3eVzb/4GL+FsXjBWpzRO4i93Sx+k+itWvJAs9frPvpiLi7OB+fK66lrv4Y9So56fVOhf11y/UbuevljcXqZ0UPdJ9JKU1OKc3vpqdbkdy67O7yBFOxr9kppVdK50X29Yi4s+Tz5oHiM6N0vRUj4pTi/H6nuD0lIlYoW6+lG9IOEfGjiHiSfJz2LtnfwRFxT0TMitz168aI2Lab3oPO+BIwi/yFstTvyJ+bn6/1iaL1i+nV1NDqMSL2AHYlJ4rmVVlt6eJ2eunMlNI7wEtATZ+BkbvkXRu5S+A7xfXx1yjpct5ynRcPv1pyznelG2dL3JWu1R2KmF4rPh8mRcRBFdb7cET8K3JX2beL1/DPiPhQsfxsWltVPl0S9zFdiLuiyH8HTy0+L96O/Pd5y7J13quBGBHfjIiHydfB4SXrfL74jH+zuBbujPxjSfn+av4bW0tsxXpLRMTxEfFk8dn3fET8JXJr3lreg8WLz9TpRUx3RcSOtWzbg3Ynt/A7qSXBBJBSupKcUKyUwG3Lt4D1gUO7K0BJXWdLJkn1cBC5e8Ofyc3lvwz8MiKmpZTOK1t3JHADcBFwCUU3hYjYrJj/GvAn4FlgY/I/HFtHxMdTSnMjf0m+jtx154/A/4BlgI3I3fvO6UxsEfEN8i/bj5J/lUvk1hqXRcSBKaXxbb0BxZeWS8jde35K7t60H/DptrYrcQ75F/h9KPmHuLA3sHjLa4uIj5N/IXyA3N3jNWBVYAdyF7P/1bjPSobEwrVJ5pd/Se5mTxa3X4+I3xWtd6qKiE+T3+u3gDOAJ4D3ATuRuyc9Waz3S/J7ehdwFLAUuWXejRGxW0rpn2VPPRrYk/xl9b3zKCJ2Af5e7Oc3wCvAVuTj/EHgc+28vhuAYyNirZRSy2vdDpgPbBQRw1NKM0rmvwHcU+W5HiF3I/htEVPLF+ryX+l3Ab4BnAacCexGPq9epfZflQeVnAuDgFXIv0ovRb5G39PN1+8tRYxHsWBX3Vq7zyxV4Rye1YMtel4gv/8fi4h1U0qP1bDNueQWJncC48jv23rAXsCPIX9xBW4jX9NnAveSu5YdDGwXuRtKeUu7E8gJmtPJ51FLLOeSu3ldTO7qNbTY/3UR8dmU0hVtBRu569jyNbyuFq+0lcQrnm9T4N6UUnlLvbvI18bmHdjfd8nv357trRgRSwMnA39KKd1VfPZXchf5uBwROYl7J7lF2r7AZuS/LbU4HLiD3I3zFfJn1AHkY7hhSullcvfzr5CP03/I531HLF5yzi9G7i53PDmJd27pihExlvy5cAf53HsL+ARwavEZ9b1ivXXJ1+rzwO/J5/n7gK3J1/Yd5Gt9aWAP8jF4qdjNez8edaNryO/TT8ldXf8P+GdENFe4Dr5TrHN6Ef8zxWv6Gbl73tXAj8jn2R7ARRFxSErplGK9jv6NbTe24rPvGvL7dzH5b8ko8vW8Y0SMrvSDSpnzyUmdK4vnWov8N+DpNrZZQPG5UtOPX9T2udlynd5eYdkdwBcjYslaWpIVybbjgGNTSlOi+8ueSeqslJKTk5NTt0zkf6YTsG+V5dsUy6cDy5bMbyT/w3V72fqTi/UPqPBc95MTPEuVzd+jNAbyl9EEHNFO7DXHBixH/pL4BLB0yfylyQmLN8ue4yZgcsnjQeSuMi8BK5bMX4ZcU6fqe1gW891FvIPK5v+neO7FiscnFs+5Ujcf71Rler6Nbfaq9fW18RxrAq8XzzOVXJfpO8BmFdZtOX4vAiMqLG8obtclf4H4fy3vWzF/VfIXhsml73PJa92h7PkWJ39JuQUYXLbsu8U227Tz+j5SrPf1knlPkb/8JWDvYl4Ur+2KkvWai3WOaWtehWVvAc0l84PcLei5Go/J5CrnwmxgbC9evzWfV7R+XlWaftHGdldR9Ezswjl8WLGfd8nJid+Tkzjvq7Du3sW657acr+Xnb3F/XLHeN8rW+WYx/7gKr/0xoLHKMRhbNn8wMJH8BTXaeX0t51WtU3M7z7dCsd7fqix/Ebitxvd+jeJ8/35ZrCdXWf9UcpfPZYrHZxfrr1hh3Y8W72npa3uD3LW31nNjiQrztq90HRTzzu7Ac7d1XJ4BPly2/irklj3nVXiu35Nbda1VPP5W8TxbtBPDMbUc8za2v6mt66/k+PyxbP7nivkHlszbppj3CmV/H8lJzQT8vMI+LiuO61LF45r+xnYwtq8X835Vtu6nivnnVjiux5TM27HS+UFOOqW23sNK73eN0zE1PN+VxbrDKiz7VbFsnRpj+yc5sTe4JNaZnTmvnJycuneyu5ykejgrpfRay4OUf/m6g9ZuUKVeIf+S/p6I2JD85fM8YGjkLiIrFr/M/j/yF4iWJuGvF7fbRsRK3RTbJ8jFf/+QSrq7FPdPIre22qGNfWwGrF7sq+WXXFJKr5N/Ma7VOeQvAZ9omRERa5B/+Tw/5W4a0Poe7Bld7/5U7s5i/6VTey11uiTlOh0bk1uSQe5G81tgYtGNY7OS1Xcid0/6TUrp2QrP1dJ6YjdyYuVXJe8bKaXp5C8GTeRWIaXuTyldXzbvE+SWcGcBy5admy0todrrrnAn+RzeDt77tXYN8q/SD5K/cAJsWLy2G9p5vlpcllKa3PIgpZTIXdDeFxELFTmuYjKt58CO5ETGneQWD/u1rNTD129n/JSFz+EzemhfAKSUfkPufnUtuavHt4C/AtMi4s8R0Viy+pji9vBU1tqn7PEe5KRjeauWP5GTzntUCOXUtHDLgy+TE+WXlR2bZclfEJup/Fld6nkWfk/bmp5v5/la3o85VZa/XbJOe04lJ8pObG/FiPgwuYvc/xWfz+2ZSb5GTwA+S26B9ARwXkR8oq0NW6Sia3FENBTdqlYkJ2VfJ9d96g6X0/re70I+/94BroiI0s+5vcit2P5cei4UMV1JLrvR8nnU8v7sFhGLd1OcXfHbssctn5OVzt2/pJReLJs3hpzwOKfCa7+C3EJzq2Ldjv6NrSW2Pcg/fBxfumJK6R/kwQ12K1r4VbN7cfvrsu0vo7XFYi0Oo/br+C9VnqNUW9fy22XrVBURXwR2Jifm3m1vfUm9y+5ykuqh0qhJL5N/rS73ZCrpt194f3F7bDFVsjJAyk2ox5FHh3ouck2bfwMXpZQqjShVS2xrFLcPVVj3weK2rSHIW5Y9WmHZw21sV+588helfcjN+SnuBwt2AzyZnET5I7nr3/8r1j8/tXa76qyXKiRaelyREDkEOCQiViG3/vkK8BngqojYIOUuey3/tP+3naes9ZhOLJlfqZthy7l5Zhv7WrmtQFLuJvb/gJb6N9uTW7zcQv4ysksxf7vitjuSTNXOe8jnfi1FcN8qPxciYgL5vT8pIq5IuatPT16/nfFAnc7hK4ErI9cHWp98nL8NfI18vA8sVh1FblHWXve/NYCJ5V+4UkrvRsRj5JYZ5aqdw0vRdnfDlats27LPt4HufE9bEmFDqyxfvGSdqiLiy+QE5sdSSnPbWXcxcvep61NK59fw3BuSuyt+N6V0Wsn8luTw6UX3smo1nVrW347cBXJL8usqtVx7cdRoWoVr9Qpy8uFU4EPF7JZrta1j2fJ5dgE5QXkUeQCGO8hdtC5IHSjI3o0W+ExLKb1cdKeq9H9GtesgqPx3ukXLa+/o39haYlsDmJ5SerXC9g+Ru16vSG7FV8ma5CRVpdf2CLn1brtSStW6YndW6bU8u2zZ4mXrVBQRy5Nrsf05pXRbt0YnqVuYZJJUD23+k12m0j8bLR3vf0NrcqXce/+YpZR+GBFnkpuZf5T86/L3IuJXKaXy0X5qia2rHf9btk9dee7iH9N/ALtH68hmXwYeSSlNLFtvc/Jr/wTwMfIvqcdGxC4ppUq1EfqNlNJz5JpdFxVJjS+REzF/pe33ulRnjmlb5+b3yL82VzK9yvxSNwA7RcQG5GTS3SmlmRFxA/CtiBhZzH+J3F2gq9o67zt9vhcJjn+Tkydbkltz9eT12+8USYcHgAeK8/dxcjHnbxTLgvbP386qdg7PIF9H1TzYxrKWwtrDOxDHjHaSL6+Sv5COqLCvoeQv5ze3E9NQclL+n8DzEbF2sajlOZcp5r1UtGb9Jrlu02El60JOwAGsERFLFy0rIXeHXZz8WfSelNKs4nP6EHIrsCepovicvpbc+ukH5BZXs8nH/wJ6cMCeIqH7KLBlRCxRtKhquVb3IXcZrOSpYvs5wCciD0KwE/nvzE+BYyLiSymlv/dU7JW0cT5V+jyrdh0k4JNU/3x8qNhXh/7G1hhbd/2f0dFlC66YEzqL1bj6zNR+LaWWv38jyOd5qRHwXtmCtvyE3Jr89LJrcxh53IK1gTkppWdqC1tSdzPJJKk/ery4nVdrC4Tii8BJ5BYVi5N/YT0iIn5ToZl8e1q+JGzAwiOcrV/cVmoZUr79+yssqzSvLeeQm8V/rmitsDb5y8kCin9qbyomIo9Kdw/wQ/KX90XFHeQvxy1fHFu6BWxCLkpbTekxLf8SWMsxbdFybi7UqqeDWlonbU9OJv25eHwT+QtPyxeZa4uubW3pqQRFrVqKxrZ8Oe+J67fer7FbpJRmRB7pbVNyK4UXyOfwbhGxcjutmZ4C1o2IwaWtmYruO+tQ2/kL+fisA9xRwxfGalanA8WFya02JldbmFKaHxH3AptExNAiodFiC3LyZWLlrd8zjJz4+hSVP/O+XEzfI3d3ayqe919Vnu8uctfOlu6kLZ85gyqsO7jstpovFdt/MqX03vsXecTF7mrF1JaWa3VJ8mtruVZrbrGaUrqL/N4QEauTWzL+jDzwAPSfa/VxcnesqSmlR9pbuQf+xj4J7BwRy5Z24S+sT64J9dJCWy24/Y7ka7m8he56HYjjUuDjNa57LLnmVlvuJg+osRULJ5m2BB6r4XOniZxkurPK8sfJr/kD7TyPpB5iTSZJ/dF/yb+kHxQRC3VLi4jBxa9vLUMFLzAyStGVo+Wfxs78434d+R/wQyOi5Yszxf1DyV2L2kpo3ANMA/aLklGtIo9iVOsIRC3+Qf5Hc59imk9uwfOeWHjkLMhdAGZTMgJU8V6tV2X9PiPysNPDKsxvIHeXg9Zuh9eS35/Dim515du0/KJ7BfnLz/dKz5dim/3IBdnb63IHOfnxIvCDlnOwbH/DSs+ZNtxLbr1xELnu1g3wXt2ue8mjES1DbV3lWv5h78hoX92iSAjtXDy8t7jtieu3bq+xoyKiMfJoVJWWjSJ/gXyJ3JoIcmF7gF+V12ApOX8hFyMeTm7pVerrxfxaW5L8hfz/4fGVFkZEm909C91dkwly9+BG8hfUUt8hdy+8sCzOtSKi9Mv0W+R6ceVTy2hxVxePW0bOO6vK+jcVy7/GgsOtt3zm7FsWx7LkrlSv0kYrpkJLC5fyliZHUfl/9pl00zkfEeuTExLPliQzLyTXzjm2ymfuMkULsWp/Z6aRz+PSGPvLtdoyyt7Pi5Z5C4iSGnG1/o3toMvIx3yBH40i4pPkH02uKK/RVuby4vZ7ZdvvTo1d5QrdXZPpcvL7ckjp+xoRnyGPfjehdOXIdbDWizzKXYtfUvnafJhc1+lz5JaFkurElkyS+p2UUoqIr5C/YE8qutI8RP4Csja54OqR5ILN2wLjI+IScouAmeTC2wcAd6bahg8v3/9rEXEEufD0nRFxdrFo32L/B7ZVJDalNC8ivkv+B/6uiDid/CXpa+Q6OCM7EMvcyDU/Dile1/UVClyfHhGrkRMuU8i/6H+e3LKk9J/CPchfrGr5NbJmkQtXf6V4uEFx+5kiJsij5HSkZsfh5GHuryQnLl4nD5W9J/k9uJGcfGvpqrI/eQjoByPiDPKvp8PJXTpOBC5PKT0WEb8GjgBuiYi/kd+fseRf9ce0V0ul2N9bEbEP+QvCY8W5+QS5aPJ65HNzD1q/qFZ7nvkRcQv5y+nb5FovLW4Avl9yv72YXo6IJ4AvFK1kXiC3tLqyvW07aJnINW8gf0lelfwlfE3g9JTS40U8PXH9PkwuVv2NiJhFHhHwxZRSd9SravkCtHHxcO1i3g+Lx6+llE7uwNM1AjdFxIPkxMbj5PdrPXKieHHgmy1fIFNKFxXn4z7AqMi1c14lJwR2ovXX+l+Rv1ydEhGbkpN5mwD7k9+7X9USXErp4og4i/wlcFPyaHovAauRWx+sTds153qiJhPk+kj7ASdGRDM50bgL+Xr6WWnLn8K/yS0eoohpLvlzYAHFc0Gu//fe8pTS/eSC2+Xrf7q4e2UqGbiBXCNmH+AXkesz3UpOMHydnCj+Zg0Fiv9O/nL8z4gYTy7G/QlyofxKrVbuAHaIiO+TR9pMKaUL2tkHwDol1+oQ8pf7A8nfC97rgppSmhYRB5ML4T8SEeeS/4YMJw88sDs5KToZ+GFE7Eg+X54mv++fIZ/XpefeHcXtLyN3D30beDCl1GYXzN6WUro7In5C/nt4X0RcRO7GtQr5M2gXWruR1fo3tiPOBr4KfL84R28hX3vfIH+GH9VO/NcUfyO/WiTtr6b1OD9Ija18ursmU9Fa80fk1oLXF/+/jCAnsx4lX0elDiF3j9uP/J5QrYt/RBwCNJVex5LqJPWBIe6cnJwWjYnWYbH3rbJ8m2rLKYb2LZs3Gbipjf01kUdjm0z+Z/xlciuh44HVi3XWKNZ5hNy8/K3i/k8phqTuTGzF/D3IX/7fKqbbqDBUNTmhMLnC/M+S6/bMIQ8ffRz5C0XV97DK+7AZrUMIj6mynyvIvyrPIf+yfDOwZ5Xjd0yN+03AVTWst01JfJWmbTp4nn2IXM/nbvI/23PJSYXbyS18hlbYZgty4uel4j2YSh7dbM2y9b5O/nL+dnG+XAd8tMprP7uNGD9AblH2bHFuvlCcHz8Clq/xdR5a7OffZfNbzpFpFbZprnQMi9d/a3Geppbzsdr6xbJjimXNNcQ6ucJxfas4RgcDDT15/Rbr7kJOOr5d7L/qZ0fZ+b5XDa/v7DbO34Wu7XaeazD5C9P55OTPG8Xrf5bcNWXbCts0kGsE3UuuH/MmMAn4Sdl6w8nFh6eRr4tp5GT4ilVee9Vrj5wY/k8R39vFcboU+HxHXm93TuRk7cnkL/tzyMnFQ4Codk7W8Jwt18DJNcbQci6sWGHZWuQuzC3v/xvk5MBnO/Aady+ug7fIn1cXkH94mFx+TpOLwl9b7Ce193pLXmvpNJ88ius1wCeqbLc1OQH2YnGuTicn8w8DFi/W2Qb4WxHn7OI57yQnhKPs+Y4gd9+cSwf+5hTb3tTW66TK3+ti2QKf27Txd79knU8V780rtP6t/hdwcMk6tf6NrTm2Yt4S5M/Dp4r3/UVyC6umKsf1mLL5w8h/K58vjsnd5MR01Th6ayJ/Bt1P/mx5kTxYxkoV1jumvWNUdm7MrOfrcnJyylOklJAkSZIkSZK6wppMkiRJkiRJ6jJrMkmS6i5qH/L8lZTSOz0dj9RREbEkrSONVTMvpTSjnXUkSZL6LZNMkqS+oNYhz7elnaLZUp0cTi5Q25Yp5PopkiRJiyRrMkmS6q4Y6v4jNax6T0rp1Z6OR+qoiFiTdkZdA2anlG7tjXgkSZLqwSSTJEmSJEmSumyR7i634oorpubm5nqHIUmSJEmStMi45557XkopLVRTdZFOMjU3NzNx4sR6hyFJkiRJkrTIiIgpleY39HYgkiRJkiRJWvQs0i2Z+rMXX4Szz4ZJk+D112GZZWCjjWC//WB4LYN8S5IkSZIk9SKTTH3M3XfD8cfDv/6VH7/9duuySy+Fn/wEPvlJOPJI2Hzz+sQoSZIkSZJUzu5yfcipp8I228Bll+XkUmmCCWD27Dzvssvyeqee2vsxSpIkSZIkVWJLpj7i1FPh8MNh1qz2100pr3f44fnxwQf3bGySJEmSJEntsSVTH3D33bUnmEq1JJocQE+SJEmSJNWbSaY+4Pjjc1e4zpg9O28vSZIkSZJUTyaZ6uzFF3OR75Q6t31K8M9/wowZ3RuXJEmSJElSR5hkqrOzz+76c0R07XkmTIDmZmhoyLcTJnQ9JkmSJEmSNLCYZKqzSZMWHkWuo2bPhltvhalT4YUX4LXX8rz589vfdsIEGDsWpkzJraKmTMmP+2OiqbeSZb2xHxN/kiRJkqT+xtHl6uz117vneS6/PE/lBg+GoUOrT5MmwZw5C24zaxYccgi89RYsu2zlabHFOhbfhAlw9NE5ETZyJIwbB2PGdOaVVn/+sWNbi6e3JMug/+2nt16LJEmSJEndySRTnS2zTPc8z9Zbw9e+lhNGpdPbby88r3yq5LXX4MADq+9v2LDqCajy6d574fe/b22xNWUKHHAAPPUUbLddjuGdd2q7rbbsqqsWLp4+axbsvz+ccUbuUtjW1NDQ/joRuX5W+SiAs2bl9+rmm/PzDBq08NSR+T/6UeV9HHYYbLQRLLEELLlknoYNy3F1Rk8n/iRJkiRJA4tJpjrbaCO45JKudZkbNgx22y0nmTqquTknfcqtvjrcfntONtUyzZgBjz/e+vjdd9ve79tvw49/nKdaReTWV4sttuDt0KHVR+ebMyd3G5w/P3cHbGuqZZ3y5E+Lt97Kia558xae5s9f8H5nvfBCPl/K35MllmhNPJUmoMrnlS67/34466zWJOOUKfD1r+f4vvKVzsdYicksSZIkSRoYInV2WLN+YPTo0WnixIn1DqNNL74ITU1dSzItvnj+Aj98eMe3Le+aBdDYCOPHdz4R0JKMaUk4bbhh5dHzIuDqqysnjSrNGzSoequdasmypiaYPLlzr6Mn9tOSzKqUgGqZNtsMnn124W1XWglOOSUntGbOzFPL/VrmVUvElVt2WVhhBVh++XxbOpXPa3m81FKVj01PnF+SJEmSpPqKiHtSSqPL59uSqc5WWgk++Um47LLKiZj2RMAuu3QuwQStX/S7s6VJaeuaESPyc1ZKzIwcCTvu2Pn9lBo3rnIyY9y47nn+7tpPRGu3uGp++cvK+zjxRNhrr87FDTmBNWtWTjqNGFH9fPvyl+Hll+GVV+Cll+Cxx/LjN96o/txDhrQmnEoTURdfXLnr31FHmWSSJEmSpEWNLZn6gLvvhm22qd4Vqy2NjbkW0OiF8od9R2+1Zumtblm9sZ+e3kdnWmTNnQuvvpoTTi1JqJb71R5XapHVYuWVYZVVYNVV823p/Zbb970vJ7DaY5c8SZIkSeo91VoymWTqI049FQ4/vGOJpsZGOOEEOPjgnouru5gE6Ft6K/FXLZm1zDKw997w3HMwfXq+feGFyjWrhg9vOxl15525ZVRvdMnzPJYkSZIkk0z9QkuiafbstrvOReRi3/0lwaS+qbdaZNWazJo3L9coK008ld5vuX3++bxue4YNy4mspZaCpZfOt+3dX2KJPOJfV1+LJEmSJC3KTDL1ExMnwvHHwz//mZNJpcWahw3LyadddoEjj+zbXeSkFt2dzJo3L9eKakk8fepT1dcdORLefDPXk6olMRWRR9+rlIS6/vpcQL3c+94Ht96aW1wtuWT14vS1srWUJEmSpL7OJFM/M2MGnH02PPBAroOz3HJ5lLZ99+18kW9pUVRLfamU8giOLQmnN9+sfr/a8kmT2o9lscXy9bniirXdrrACDC4ZfsHWUpIkSZL6A5NMkhZJ9a4vNXw4/OpXuXXVjBmVb197rfrzLrdcTjqtuCLcd9+CrRdbrLoqPPJIblHV1ZZSkiRJktRV1ZJMgyutLEn9RUsiqae7mI0bVzmZ9dvftr+vuXNzsqmtRNSMGZUTTJC7BS6zTN5fS/HzStP73pdvV1ihem0psEueJEmSpJ5hSyZJqlFPJ2eqtZZaYQX4/vdbi6GXTm++ufD6Q4bAyitXTkQ99hj88Y+5+2ALu+RJkiRJ6gi7y0lSH9eZrn9vvdWacHr++cqJqOeey62l2rLMMnD66bDBBjBqVE5USZIkSVIldpeTpD6uM13/llgC1l47T2155x144YVcEL3Sbwuvvw57753vDxkC66yTE07rr59vN9gg78PkkyRJkqRqbMkkSQNItS55q68Ol18ODz204PT0061JqSFDYN11W5NOLdNaay04Sl4Laz9JkiRJiyZbMkmSqhYwP/542GSTPJWaNSuPbPfQQ/Dww/n2rrvgb39rXWexxRZOPj31FPz4x637mTIl7xdMNEmSJEmLKpNMkjSAdLRLXmMjbLZZnkq99VZr8qlluv12uOCC6vueNQu+9S1YbrlchHzVVWH48LZHwpMkSZLUf9hdTpLUbWbOzC2ettyytvUHD4b3vS8nnFZdtTX5VDqtskoeYa9SMsoueZIkSVLv6xPd5SJiZ+D3wCDgjJTSL6qstzlwB/D5lNLFJfMHAROBZ1NKn+6FkCVJHbDkkrDFFrnAeKXaTyNGwMUXw/TpeXruudb7TzwB//kPvPzywtsNGbJgAmqVVeDFF3MdqXfeyevYJU+SJEmqr15LMhUJolOATwDTgLsj4oqU0sMV1vslcE2Fp/k28AiwdA+HK0nqgmq1n375S/jQh9re9u234fnnKyeipk+HRx+FG2+EV19deNtZs+CrX4UTT8wtpEqnVVZZ8PGSS9b2WmwtJUmSJNWmN1sybQE8kVJ6CiAiLgB2Ax4uW+9Q4BJg89KZEbEa8ClgHPB/PR6tJKnTOlr7qdTii+dR8Jqb216voaF15LtS8+blJNLzz8N998ELL+R55ZZYYuFEVPl0++3wgx9YwFySJEmqRW8mmUYAz5Q8ngYsULUjIkYAewDbUZZkAn4HHAEs1XMhSpK6y5gxPZuIGTmycpe8pib4xz9aH8+fn7vgPf98bhX1/PMLTw8/DDfcULl1VLlZs+Cb38y3q63WOi27LER028uTJEmS+p3eTDJV+te7/Dfo3wHfTynNi5L/1CPi08CLKaV7ImKbNncSMRYYCzBy5MguhCtJ6suqdckbN27B9Roa8ih2w4fDhhu2/Zxvv51rPbUko3bfvfJ6r7/e2qKpdN8jRiyYeFpttQXntTWant3yJEmS1N/1ZpJpGrB6yePVgOll64wGLigSTCsCu0TEu+QWT7tGxC7A4sDSEfHXlNKXy3eSUhoPjIc8uly3vwpJUp/QlS551Sy+eH6elt8oqhUwHzkSbrkFnn0Wpk1rnVoe33xzrh/17rsLbjdkSE46lSejnnoKxo+HOXPyenbLkyRJUn8UqVJBi57YUcRg4H/A9sCzwN3Al1JKD1VZ/2zgqtLR5Yr52wCH1zK63OjRo9PEiRO7FrgkacCaMKFya6nx49tP/syfn1tFlSegyh/Pnl39OVZeOSegGhu75/VIkiRJ3SEi7kkpjS6f32stmVJK70bEIeRR4wYBZ6aUHoqIg4rlp/VWLJIk1aIrraUaGloLiI9e6M9vllKuA7XiipWLmL/wAiyzDHzwg7D11vDhD+fbESM6/ZIkSZKkHtNrLZnqwZZMkqT+oLm5cre84cPhgAPgttvgrrtaWz2NHJmTTS2Jpw03hMG92QFekiRJA1q1lkxVyo9KkqTeMm7cwl3iGhvht7+Fn/8cbropFxu/6648b8stc92nQw6BTTeF5ZaDHXaAH/8Yrrkmr1vJhAk5odXQkG8nTOjhFyZJkqQBxZZMkiT1AR0dXS6lvO5tt8Gtt+bb++/PtaAi4AMfaO1e9+EPwx13dL6+lCRJklSqWksmk0ySJC0i3nwzt3ZqSTrdfju88UZe1tCQE1Dlmppg8uReDVOSJEn9XN0Lf0uSpJ611FKw/fZ5Apg3Dx5+OCedDj648jZTpsAGG+Ri4quumm/Lp5VWgkGDaouhoy2yJEmStOgwySRJ0iJq0KBcFHzDDeEXv6hcXHyppWDddeHZZ3NC6vnnc3Kq/HlWWaU16VQtGXX55Qt2yZsyJT8GE02SJEkDgd3lJEkaACZMqK0m07x58MILOen07LMwfXrr/dKppRteqYhcK6rcaqvBM890/2uSJElSfdhdTpKkAawlkdReV7ZBg3JLpVVXhc03r/58M2cunHg68sjK606bBquvDh/84ILTGmvkWlGSJElaNNiSSZIkdYvm5spd8pZdFnbZBe67Dx59tLUA+VJLwcYb56kl8bTBBjBsWK+FLEmSpE6o1pLJ3w8lSVK3GDcud8Er1dgIJ5+cu+s99FBuAXX33XD66bDPPrl73TnnwNe/nltOLbUUfOAD8OUvw69/DdddBzNmLLyvCRNyUquhId9OmNAbr1CSJEltsbucJEnqFrV0yRs2DEaPzlOL+fPh6adzS6eW6ZZbFkwcrbpqa2unN96AM86At9/OyywwLkmS1DfYXU6SJPVJL78M99+/YPLpkUfg3Xcrr7/KKjm5Ndif0CRJknqUhb8lSVK/ssIKsN12eWoxZ05uDVXpN7Lnnmut87TZZnnadNNc52nIkN6LW5IkaaAyySRJkvqNoUNzN7xKBcZXWCHXebr3Xjj3XPjjH1u32WijnHBqSTx94AN5viRJkrqPhb8lSVK/Uq3A+O9/DyeeCDfdBK+9Bo89BuefD4ceCksuCRdckGs3jR6dWzxttlkuOH7aabkYeUuNpxYWF5ckSeoYazJJkqR+Z8KEtguMV5ISPPVUbul0zz2t06uv5uWDB+eudZtumouR/+1vCyaeGhth/HiLi0uSJFWryWSSSZIkDVgp5a5399yzYPLppZcqr7/qqvDMM7l1kyRJ0kBl4W9JkqQyEbkrXHMz7LlnnpcSDBpUubj49Omw4orwoQ/BVlvlaYstYOmlezNqSZKkvskkkyRJUomI6sXFl18e9tgD7rgDrr46J6Iicje7lqTTVlvBOuvY2kmSJA08JpkkSZLKjBuXi4TPmtU6r7ER/vCH1ppMr70Gd90Ft9+epwsvhNNPz8uWW87WTpIkaeAxySRJklSmJZHUVnHxZZeFHXfME+Ri4Y891pp0uv322lo7daaIuSRJUl9k4W9JkqQe8vrrcOeduXvd7bfn29dey8uWWw5WXx0eeQTmzm3dxlHsJElSX+focpIkSXVW3trpnHMWTDC1GD48r7fccr0foyRJUntMMkmSJPUxDQ2VR7GD3MVuo43gYx/L00c/Ciuv3LvxSZIkVVItyeS4J5IkSXUycmTl+SuvDMcem1s0/fnP8LnPwfveB+utBwcemOs4PfNM78YqSZLUHgt/S5Ik1Um1Uex+85vWmkxz58K998LNN8Mtt8Df/pZrNgE0N+dWTh//eL5da63cAkqSJKkebMkkSZJUJ2PG5IRRU1NODjU1LVz0e8gQ2HJLOOIIuOoqePll+O9/4fe/h003hX/+E/bfH0aNghEj4AtfgD/+ER56KNeAgtzyqbk5d89rbs6PJUmSups1mSRJkvqxlODRR3Mrp1tuyS2enn02L1thhZxUmjTJEewkSVL3sfC3JEnSAJASPP10a9Lp3HPh3XcXXm/4cHjwQVhppd6PUZIk9W8mmSRJkgagtkawA1hzTfjQh2CrrfLtxhvnLnqSJEnVVEsyWfhbkiRpETZyJEyZsvD8lVeGww+H22+HG2+E887L8xdfHEaPzgmnluTTqqv2bsySJKl/qrnwd0RsGBEnR8S/ImKVYt7uEbFJz4UnSZKkrhg3LtdgKtUygt3hh8Mll+QaTlOn5pHrDj4Y5s2DP/wB9torFxMfORL23ht++9uclHr77YX3Y3FxSZJUU0umiNgRuAL4F7AdMKxYtBawL7B7D8QmSZKkLmop7n300TmRNHJkTjyVFv2OgNVXz9Pee+d5c+bAffflpNIdd+TpoovyssUWg002aW3t9MILcNRRMGtWXj5lCowdu+D+JUnSoq+mmkwRcSdwTkrpjxHxJrBxSumpiNgMuDKl1CcbUVuTSZIkqftMnw533pkTTrffDhMnwuzZ1ddvaoLJk3stPEmS1Euq1WSqtbvcBsA/K8x/BVi+K4FJkiSpf1h1VdhjD/jlL/PIda+/DvfcU339KVPgzDPhmWd6L0ZJklQ/tSaZXgVGVJi/KTCt+8KRJElSfzFkCGy6aW6xVElDA+y/f+6i9/73w7e+BVdeCW++2btxSpKk3lFrkuk84NcRsRqQgMER8XHgBOAvPRWcJEmS+r5qxcX/8he4/3444YSciDrjDNh1V1h+efjYx+C443LXu3ffrU/ckiSpe9Vak2kIcDbwBSCA+cXtecC+KaV5PRhjp1mTSZIkqXdMmNB2cXHIo9Lddhtcdx1cey3897+QEiyzDGy3Hey4I3ziE7DWWvV5DZIkqTbVajK1m2SKiAZgPWAqsBK5i1wD8N+U0uM9EGu3MckkSZLUd730Evz73znpdN11OUEFsMYaOdn0iU/k5NPyJRVAa0lmSZKkntWVJFMAc4D1U0pP9FB8PcIkkyRJUv+QEjz+eGsrpxtvzLWbGhpg9OiccAL47W9h1qzW7RobYfx4E02SJPWmTieZio0fAMamlG7vieB6ikkmSZKk/mnuXLjrrtZWTnfeCfOqFGhoaoLJk3s1PEmSBrRqSaZaC38fQS78/cGiZZMkSZLUY4YMga23hmOOgVtvhZdfhmr/hU6ZAmee2drdTpIk1cfgGte7EFgcuAd4NyLmlC5MKS3d3YFJkiRJLZZZJtdgmjJl4WUNDbD//vn+qFGw/fawww6w7bYL1nOSJEk9q9Yk0yE9GoUkSZLUjnHjYOzYyjWZNt44FxG//nr461/htNNyy6dNN80Jpx12yC2jhg2rX/ySJC3qaqrJ1F9Zk0mSJGnRUsvocnPnwt1354TT9dfDHXfkeUOH5kTTDjvk1k6bbQaDBtXndUiS1J91qfB38QRDgTHA+kACHgLOTynNaXPDOjLJJEmSpJkz4T//aW3pdP/9ef6yy8I227S2dFpnndz6qZZEliRJA1lXR5dbH7gaWBp4oJi9IfA6sHNK6ZFujLXbmGSSJElSuRdfhBtvbG3p1DIy3YgRsOaaeSS7d95pXb+lS56JJkmSsq4mma4DZgFfSSm9UcxbGvgrMDSltFM3x9stTDJJkiSpPU891ZpwuuQSmD9/4XWqFR2XJGkgqpZkaqhx+62Bo1oSTADF/aOBj3QgiJ0j4rGIeCIiftDGeptHxLyI2Kt4vHpE3BgRj0TEQxHx7Vr3KUmSJLVlzTVzQfELL4Rqv79OnQqf/jScfDI88UTvxidJUn9R6+hybwPLVpi/TLGsXRExCDgF+AQwDbg7Iq5IKT1cYb1fAteUzH4XOCyldG9ELAXcExHXlW8rSZIkdUW1FktLLQWPPQb/+Ed+vNZasPPOedp2W1hiid6NU5KkvqjWlkxXAqdHxNYRMaiYPgL8CbiixufYAngipfRUSukd4AJgtwrrHQpcArzYMiOl9FxK6d7i/pvAI8CIGvcrSZIk1WTcuFyDqVRjI5x6Kjz+eJ5OPhne/3446yz4zGdg+eVz4fATToAHH6zeGkqSpEVdrUmmbwOPA/8ht1x6G7gZ+B/wnRqfYwTwTMnjaZQliiJiBLAHcFq1J4mIZmAT4M4a9ytJkiTVZMyYXOS7qSmPNNfUtGDR77XXhm9+E668El55JY9Y9+1v52Li3/sebLghrL46HHAAXHwxvPZaXV+OJEm9qqYkU0rptZTSbsA6wGeBPYF1U0p7pJRer3FfUempyx7/Dvh+SmlexSeIWJLcyuk7pfWhytYZGxETI2LijBkzagxNkiRJysaMySPOzZ+fb6uNKjd0KGy3HfzqVzBpEjzzDJxxBmy1VU4wfe5zsOKK8JGPwM9+BhMnthYVnzABmpuhoSHfTpjQO69NkqSeVOvocosBDSmlt8vmLw7ML7q/tfccWwHHtIxEFxFHAqSUji9Z52lak1Erkke0G5tSuiwihgBXAdeklE6s5cU5upwkSZLq4d134c474Zpr4Oqrc4IppZx0Wmed/Pidkv+gGxsXbDElSVJfVm10uVqTTJcDN5cndyLiO8A2KaXda3iOweTuddsDzwJ3A19KKT1UZf2zgatSShdHRADnAK+klL7TbsAFk0ySJEnqC2bMgOuuywmnCRNaWzSVamrKLackSerrqiWZaq3JtDVwbYX51wEfruUJUkrvAoeQR417BLgwpfRQRBwUEQfVsP+vANtFxH3FtEuNsUuSJEl1NXw4fOlL8Je/VC8MPmUKHH88PPFE78YmSVJ3qbUl0yxg05TSo2Xz3w/cm1Ia1kPxdYktmSRJktTXNDfnhFK5xRZr7UK3ySaw9965rtNaa/VqeJIktaurLZkmAV+sMP9LwINdCUySJEkaSMaNyzWYSjU2wpln5uTTb36Ti4ofeWQezW6zzeAXv4Ann6xPvJIk1arWlkyfAi4DLgRuKGZvD3wO2COldFVPBdgVtmSSJElSXzRhAhx9NEydCiNH5sRTedHvqVPzKHUXXpiLiANsumlu3WQLJ0lSPXWp8HfxBDsDPwQ2KWb9FxiXUvpXt0XZzUwySZIkaVEwZUpOOF10kQknSVL9dbW7HCmlq1NKH0kpLVFMH+nLCSZJkiRpUdHUBIcdBnfckUegO+EEGDJkwS51v/wlPPXUgttNmJBrQDU05NsJE+oQvCRpwKi5JdN7G0QsDuwNLAFcl1Lqs+Nf2JJJkiRJi7JKLZw22yy3blpsMfjhD2HWrNb1Gxth/PiFu+ZJktQRneouFxE/BRpTSocXjwcDdwEfLFZ5C/hESumObo+4G5hkkiRJ0kDRknC68EK4667q6zU15dZQkiR1Vme7y+0G3F7y+IvAesBHgBWBm4GjuitISZIkSZ3T0qXuzjvh6aerrzd1au/FJEkaWNpLMjUBD5U83hG4JKV0W0rpFeBnwGY9FZwkSZKkjmtuzkmnShoa4LjjYPr0Xg1JkjQAtJdkGgS8U/J4S+C2ksfTgeW7OyhJkiRJXTNuXK7BVGqxxeD974cf/xhGjoQ99oCrr4Z58+oToyRp0dJekulxYDuAiFgDWIvcRa7FasBLPROaJEmSpM4aMyYX+W5qgoh8e+aZ8MAD8MQTcPjhcOut8MlPwlpr5aTUc8/VO2pJUn/WXuHvA4DfA5cAWwAvp5S2Lln+Q2CLlNKuPR1oZ1j4W5IkSarunXfgssvgT3+CG26AwYNh113hwANhhx1y1zpJksp1qvB3SukM4FBgKeBGYM+yVVYFzuyuICVJkiT1nsUWg733hn//Gx57DL7zHbjlFthpJxg1Cn7xC3jhhXpHKUnqL9psydTf2ZJJkiRJ6pg5c+DSS3PrpptvhiFDYPfdc+umbbe1dZMkqZMtmSRJkiQNLEOHwhe/CDfdBI88Aocckls67bADrLsu/PrXMGNGvaOUJPVFJpkkSZIkVbTeenDiifDss3DuubDKKnDEEbDaaq2JqAkToLk5t3Bqbs6PJUkDk93lJEmSJNXs4YfzqHXnnAOvvZZHriv9StHYmJePGVO3ECVJPaxadzmTTJIkSZI6bPZsWH11ePnlhZeNHAlTpvR+TJKk3mFNJkmSJEndZtgweOWVysumToWTT4bXX+/dmCRJ9dWlJFNErB4RZ3ZXMJIkSZL6j5EjK89fbDE49FBYdVU44ACwc4EkDQxdbcm0PPDV7ghEkiRJUv8yblyuwVSqsRHOPBPuvhu+9CU4/3zYfHMYPRrOOAPeeqs+sUqSel6bNZkiYp92th8JHJtSGtStUXUTazJJkiRJPWvCBDj66NxFbuTInHgqLfr9+uvw17/CaafBgw/C0kvDV74CBx4IG25Yv7glSZ3XqcLfETEfmAVUW6kBWNwkkyRJkqS2pAS3356TTRdeCHPmwNZbw0EHwV57weKL1ztCSVKtOlv4ezqwT0ppqUoTsHWPRCtJkiRpkRIBH/4w/OUv8Oyz8JvfwIsv5lZNI0bAYYfB//5X7yglSV3RXpLpHmDTNpYnILovHEmSJEmLuhVWgP/7P3jsMfj3v2H77eEPf4B11833L7oI3nmn3lFKkjqqvSTTCcCtbSx/Ati2+8KRJEmSNFBEwHbb5e5zzzyT6zk9+STsvXeu73T00TB5cq771NwMDQ35dsKEOgcuSaqozZpM/Z01mSRJkqT+Zd48uPbaXLvpqqtg/vycXJo/v3WdxkYYP37BAuOSpN7TqZpMEbFRRLTX2kmSJEmSusWgQfDJT8Lll+dWTMsss2CCCWDWLDjqqLqEJ0lqQ3sJpP8CK7Y8iIh/RMQqPRuSJEmSJMHqq8Mbb1ReNnUqHHccPPFE78YkSaquvSRTeVHvjwHDeigWSZIkSVrAyJGV5w8dCj/+MYwaBZtvDieemEetkyTVj13hJEmSJPVZ48blGkylGhvhz3/OxcJPOAFSgsMOyy2ftt0212t6+eX6xCtJA1l7SaZUTOXzJEmSJKnHjRmTk0ZNTXk0uqam1qLfq62Wk0sTJ8Jjj8FPfgLPPQcHHgjvex98+tN5JLqZM+v9KiRpYGhzdLmImA9cB8wpZn0SuBmYVbpeSmnXngqwKxxdTpIkSRpYUoL77oPzz8/TtGkwbBjsuit88Yuw8865q50kqfOqjS7XXpLprFqePKW0Xxdi6zEmmSRJkqSBa/58uPXWnGy66CJ46SVYdln47GdzwmnbbfNodpKkjulUkqm/M8kkSZIkCWDuXPj3v+G88+Dvf89d6FZeGT7/+Zxw2nLLvOzoo/PIdSNH5npQY8bUO3JJ6ntMMkmSJEkSMHs2/OMfuYXTP/4Bc+bAiivCa6/Bu++2rtfY2Fr/SZLUqlqSydHlJEmSJA0ow4bBXnvBJZfACy/A2Wfnlk2lCSaAWbNyyyZJUm1MMkmSJEkasJZZBr761dyaqZIpU/IkSWqfSSZJkiRJA97IkdWXrbkm7LEH3HBDHr1OklSZSSZJkiRJA964cbkGU6nGRvj97+GII+A//4Htt4cNNoA//hHefLM+cUpSX1ZzkikiGiPiwxGxe0R8tnTqyQAlSZIkqaeNGZOLfDc1QUS+HT8evvUtOP54mDYt125qbIRvfhNWWw2+/W147LF6Ry5JfUdNo8tFxA7A+cAKFRanlNKg7g6sOzi6nCRJkqTulBLceSecfDJceCHMnQs77giHHAK77AKD+uQ3I0nqXl0dXe73wD+A1VJKDWWTH6OSJEmSBoQI+NCH4K9/hWeegeOOgwcfhF13hVGj4IQT4JVX6h2lJNVHrUmmZuC4lNL0HoxFkiRJkvqNlVeGH/4QJk/OrZpWXx2+9z0YMQIOOADuu6/eEUpS76o1yXQrsG5PBiJJkiRJ/dGQIfC5z8HNN8P998M++8B558Emm8BHPwp/+1vuVgcwYQI0N0NDQ76dMKGekUtS96q1JtNngZ8BJwIPAHNLl6eU7u2R6LrImkySJEmS6uHVV+Gss+CUU+Cpp2CVVWCrreBf/4LZs1vXa2zMBcbHjKlfrJLUUdVqMtWaZJrfxmILf0uSJElSBfPn58TSySfD1VdXXqepKXe5k6T+olqSaXCN26/RzfFIkiRJ0iKvoQE+9ak8NTTk0enKTZ3a+3FJUk+oqSZTSmlKW1OtO4uInSPisYh4IiJ+0MZ6m0fEvIjYq6PbSpIkSVJfNHJk5fkNDXDSSTBzZu/GI0ndrdbC30TERhHxl4iYGBF3R8Q5EbFhB7YfBJwCfBJYH/hiRKxfZb1fAtd0dFtJkiRJ6qvGjcs1mEoNHQprrAHf+lYene7II2G6Y3pL6qdqSjJFxK7AvcDqwL+Aq4GRwL0R8Zka97UF8ERK6amU0jvABcBuFdY7FLgEeLET20qSJElSnzRmTC7y3dQEEfn2z3+Gxx+H226D7beHX/0qjzq3774waVK9I5akjqm1JdPPgHEppW1TSj8qpm2B44tltRgBPFPyeFox7z0RMQLYAzito9tKkiRJUl83Zkwu8j1/fr5tGVVuq63g4ovhf/+DAw+Eiy6CjTeGHXeEa66pXMtJkvqaWpNM6wDnVph/LrBujc8RFeaVf1T+Dvh+SmleJ7bNK0aMLbr0TZwxY0aNoUmSJElS/a21Vq7P9Mwz8POfw4MPws4754TT2WfDnDn1jlCSqqs1yfQisFmF+ZsBL9T4HNPI3e1arAaU9zYeDVwQEZOBvYA/RsTuNW4LQEppfEppdEpp9PDhw2sMTZIkSZL6juWXz/WZnn46J5cA9tsvd6X7+c/hlVfqGZ0kVVZrkul04E8RcXREbBsR20TED8nd2sbX+Bx3A6MiYo2IWAz4AnBF6QoppTVSSs0ppWbgYuAbKaXLatlWkiRJkhY1Q4fCV78K99+fu81ttBEcfXQuEn7oofDkk/WOUJJadaQm07HAwcC/gRuAg4CfAD+v5QlSSu8Ch5BHjXsEuDCl9FBEHBQRB3Vm2xpjlyRJkqR+LaK1PtOkSbD33vCnP8GoUbDnnrlwuCTVW6QOVpCLiKUAUkpv9khE3Wj06NFp4sSJ9Q5DkiRJkrrd9Olw8slw2mnw6qu5ePhhh8Huu8MFF+QWT1OnwsiRMG5ca5FxSeqqiLgnpTR6ofkdTTL1JyaZJEmSJC3q3noLzjoLfvtbeOopGD4cXnsN5s5tXaexEcaPN9EkqXt0OMkUEZOAj6eUXo2IB6gymhtASmmjbou0G5lkkiRJkjRQzJsHl18OX/pS5VHomppg8uReD0vSIqhakmlwG9tcAswpub/oNnmSJEmSpH5u0CD47GfhnXcqL58yJbdwWnbZ3oxK0kBSNcmUUjq25P4xvRKNJEmSJKlLRo7MCaVKVlklFwr/2tdgm22godahoCSpBjV9pETEDRGxbIX5S0fEDd0elSRJkiSpU8aNyzWYSjU2ws9+lpNLV10F228Pa68Nxx0HzzxTnzglLXpqzVtvAyxWYf7iwEe7LRpJkiRJUpeMGZOLfDc1QUS+HT8+jzZ3yinw3HMwYQKsuSb8+Md5+c47w0UXVa7lJEm1anN0uYjYtLg7EdgReKVk8SBgJ+CAlFJzTwXYFRb+liRJkqTqnn46j0x39tm5RdMKK8CXv5xbPG3UJ4d3ktQXdHh0uWKj+bQW/I4Kq8wGDk0pndktUXYzk0ySJEmS1L558+D66+HMM+Gyy3Lx8NGjc7Lpi1+0WLikBVVLMrXXXW4NYC1ygmmL4nHLNAJYuq8mmCRJkiRJtRk0CHbaCf72N5g+HX7/+5xo+sY3crHwL38ZbrwR5s+vd6SS+rI2k0wppSkppckppYaU0sTiccv0XEppXm8FKkmSJEnqeSusAN/6Ftx3H0ycCPvtl4uFb7cdjBqVC4g/80yu69TcnEeoa27OjyUNbG12l1tgxYjB5NZMIykrAp5S+kv3h9Z1dpeTJEmSpK6bPRsuvTR3p7uhGF+8oWHBlk2NjbnA+Jgx9YlRUu/pVE2mko3XA64kd5MLYB4wGJgLzEkpLd294XYPk0ySJEmS1L2eego23RRef33hZU1NMHlyr4ckqZd1tiZTi98B9wDLALOA9wOjgfuAPbsnREmSJElSX7fmmvDGG5WXTZkC55wDs2b1bkyS+oZak0ybAz9LKb0FzAcGp5TuBY4AftNTwUmSJEmS+p6RIyvPHzwY9t03Fwv/xjfgv//t1bAk1VmtSaYgt2ACmEEeWQ5gGrB2dwclSZIkSeq7xo3LNZhKNTbC2WfDzTfDrrvCWWflbnWbbQannVa5e52kRUutSaYHgY2L+3cB34+IjwPHAk/0RGCSJEmSpL5pzJhc5LupCSLybUvR7499DM49F6ZPh5NOgnffhYMPhlVXzSPV3XYb1Dj+lKR+ptbC3zsBS6SULo2INYGrgPWAl4C9U0o39WiUnWThb0mSJEmqr5Rg4kQ4/XQ4/3yYORPWXx8OOAC+8hVYccV6Ryipo7o0ulyVJ1weeDV19gl6gUkmSZIkSeo7Zs6Ev/0NzjgD7rgDFlsM9tgDvv512HZbaKi1r42kuurq6HILSSm90pcTTJIkSZKkvmXJJWH//eH222HSJDjoILj2WthhBxg1Cn7+89zNTlL/VLUlU0TcCNSUREopbdedQXUXWzJJkiRJUt/29ttw6aW5ddONN8KgQfCpT+XWTa++Cj/6EUydmke0Gzcu132SVF/VWjINbmObB0vuDwLGAM8DdxbztgBWAf7aXUFKkiRJkgaWxReHL30pT48/Dn/+cx6l7oorclHxlnYRU6bA2LH5vokmqW+qtfD3b8mJpm+XdpGLiN8Vz/HtHouwC2zJJEmSJEn9z9y5MGIEzJix8LKmJpg8uddDklSiqzWZ9gFOrlCD6Y/AV7oanCRJkiRJLYYMgZdeqrxsyhS44YbWFk6S+o5ak0wBbFhhfqV5kiRJkiR1yciRlec3NMD228MWW8All8C8eb0bl6Tqak0ynQmcERE/iIhtiukHwOnAWT0XniRJkiRpIBo3DhobF5zX2JhrNv3pT7ko+F57wfrr56Lhc+bUJ05JrWpNMh0BHA8cCtxQTIcCvyiWSZIkSZLUbcaMgfHjcw2miHw7fjzsu28uAP7YY3DhhbDkknkkujXWgF//Gt54o96RSwNXTYW/F9ggYmmAlFKfv3Qt/C1JkiRJi7aU4N//hl/8It8uswx84xvw7W/DyivXOzpp0dTVwt/vSSm90R8STJIkSZKkRV8E7LADXH893H03fOITOeHU1AQHHwxPPlnvCKWBo2qSKSImRcRyxf0HiscVp94LV5IkSZKkykaPhosugkcfhX32gTPPhHXWgS9+Ee67r97RSYu+wW0suwRoKZ12cS/EIkmSJElSl62zTq7fdOyx8LvfwamnwgUXwE47wfe/D9tsk1tASepeHa7J1J9Yk0mSJEmS9NprcNppOeH0wguwxRY52bT77tDQABMmwNFHw9SpMHJkHtluzJg6By31YdVqMplkkiRJkiQNCG+/Deeck0ehe/JJWHdd+OhH4bzzYNas1vUaG3NLKBNNUmUdTjJFxANATRmolNJGXQuvZ5hkkiRJkiSVmzcPLrkkFwj/738rr9PUBJMn92pYUr9RLcnUVk0m6zBJkiRJkhY5gwbB3nvD5z6X71dqezF1au/HJfV3VZNMKaVjezMQSZIkSZJ6U0SuwTRlysLLFl8cbroJPv5xi4RLtWqodwCSJEmSJNXLuHG5BlOpIUNyC6dtt4VNNoGzz871nCS1reYkU0TsFxHXRsSjEfFU6dSTAUqSJEmS1FPGjMlFvpuacoulpiY46yx48UU444xcv2m//fL8Y47Jo9NJqqymJFNEfA/4DXAP0AxcBjwILA+c2UOxSZIkSZLU48aMyUW+58/Pt2PGwLBhsP/+MGkSXH89bLEFHHts7l63777VC4ZLA1mtLZm+DoxNKR0JzAVOTintSk48NfVUcJIkSZIk1VMEbL89XHklPPYYjB0LF18Mm24K22wDl12WWztJqj3JtBpwV3F/NrB0cf98YM/uDkqSJEmSpL5mnXXgpJNg2jQ44YTc6mmPPWDUKPjd7+CNN+odoVRftSaZngdWLO5PAbYq7q8NVBjsUZIkSZKkRdOyy8Jhh8ETT+RWTSNGwHe/C6utBt/5Djz5ZL0jlOqj1iTTDcCuxf0/AydGxI3A34BLeyIwSZIkSZL6ssGDYc894T//gbvvht12g1NOyS2bdt8dbroJks0yNIC0mWSKiO2Lu2OBnwGklE4D9gUeAI4GvtGD8UmSJEmS1OeNHg3nngtTpsDRR8Ott8K228Imm8DZZ+epuRkaGvLthAn1jVfqCZHaSKtGxHxgMrn10lkppem9FFe3GD16dJo4cWK9w5AkSZIkDTCzZ8N55+VaTQ8+uPDyxkYYPz6PZCf1NxFxT0ppdPn89rrLbUDuDncoMCUi/hERu0fEoJ4IUpIkSZKkRcGwYbD//jBpEqy00sLLZ83KLZ6kRUmbSaaU0iMppcPJo8t9nlzk+yLg2Yj4ZUSs2wsxSpIkSZLUL0XAjBmVl02ZAvff37vxSD2ppsLfKaV3U0qXppQ+DTQBfwA+CzwcEbf0ZICSJEmSJPVnI0dWnh8BH/wg7LIL3HKLRcLV/9U6utx7irpMfyQnml4Dtq5124jYOSIei4gnIuIHFZbvFhGTIuK+iJgYER8pWfbdiHgoIh6MiPMjYvGOxi5JkiRJUm8bNy7XYCrV2AinnQY/+xlMnAgf/zhsvTVccQXMn1+fOKWu6lCSKSJ2iIjzgOnAscAFwEKFnqpsOwg4BfgksD7wxYhYv2y1fwMbp5Q+CHwNOKPYdgTwLWB0SukDwCDgCx2JXZIkSZKkehgzJhf5bmrKrZeamvLjsWNzXabJk+Hkk2H6dNhtN9hoozxS3dy59Y5c6ph2k0wRMTIifhIRTwPXAqsCY4FVU0rfTCn9t8Z9bQE8kVJ6KqX0DjlBtVvpCimlmal1uLslyDWgWgwGhkXEYKCRnOiSJEmSJKnPGzMmJ5Pmz8+3paPKNTbCN78Jjz8Of/1rTkTtsw+svTacdFIuEi71B20mmSLiOuAp4EByUmidlNI2KaW/ppTe7uC+RgDPlDyeVswr3+ceEfEo8A9yayZSSs8CJwBTgeeA11NK13Zw/5IkSZIk9VlDhuTk06RJcOWVsNpq8K1v5ZZPP/sZvPpqvSOU2tZeS6bZ5ALfq6eUjkwpPdGFfUWFeQuVNUsp/T2ltB6wO3AcQEQsR271tAa5JdUSEfHlijuJGFvUc5o4o1oJf0mSJEmS+qgI+PSn4dZb4T//gS23hB/9KBcQP/xwePbZekcoVdZmkimltGtK6YqU0rxu2Nc0YPWSx6vRRpe3lNItwFoRsSKwA/B0SmlGSmkucCnw4SrbjU8pjU4pjR4+fHg3hC1JkiRJUn185CNw1VVw//25XtPvfgdrrAEHHAD/+1+9o5MW1OHR5brgbmBURKwREYuRC3dfUbpCRKwdEVHc3xRYDHiZ3E3uQxHRWCzfHnikF2OXJEmSJKluNtoo12v63//g61+HCRNgvfXgc5+De+7J60yYAM3N0NCQbydMqGfEGoh6LcmUUnoXOAS4hpwgujCl9FBEHBQRBxWr7Qk8GBH3kUei+3zK7gQuBu4FHijiHt9bsUuSJEmS1BesuSacckouHn7kkXDddTB6NHzgA7D//jBlCqSUb8eONdGk3hWtg7ktekaPHp0mTpxY7zAkSZIkSeoRr78Of/pTTjjNn7/w8qamnJCSulNE3JNSGl0+vze7y0mSJEmSpG60zDJwxBG59VIlU6f2bjwa2EwySZIkSZLUz40cWXl+BBx9NDz3XO/Go4HJJJMkSZIkSf3cuHHQ2LjgvKFDYdNN4fjjcyHw/faDBx6oS3gaIEwySZIkSZLUz40ZA+PH5xpMEfn2z3+Gu+9uHZHuwgvzKHU77QTXXlu9i53UWRb+liRJkiRpAHjlFTjtNDjpJHj+edhwQ/i//4MvfjG3epJqZeFvSZIkSZIGsOWXh6OOyqPNnXVWbsm0336wxhrw85/nJJTUFSaZJEmSJEkaQIYOhX33hUmT4Oqrc4umo4+G1VeHQw6BJ5+sd4Tqr0wySZIkSZI0AEXk+kzXXAP33w+f+1yu6zRqFOy5J9x2W70jVH9jkkmSJEmSpAFuo43g7LNzV7of/ABuvBG23hq22gouvhjmzat3hOoPTDJJkiRJkiQAVl0112d65plcIPzFF3MLp1Gj4A9/gJkzYcIEaG6GhoZ8O2FCvaNWX+HocpIkSZIkqaJ58+Dyy+E3v8nd54YNg7lz4d13W9dpbMzd7MaMqV+c6l2OLidJkiRJkjpk0CD47Gfh1lvh9ttzHafSBBPArFm5cLhkkkmSJEmSJLXrQx+C2bMrL5s6tXdjUd9kkkmSJEmSJNVk5MjK81OCvfeGSZN6Nx71LSaZJEmSJElSTcaNyzWYSg0bBrvuCldfDRtvDLvvDvfcU5fwVGcmmSRJkiRJUk3GjMlFvpuacn2mpiY4/fRcHHzKFDjmGLj5Zhg9GnbZJddx0sDh6HKSJEmSJKnbvPEGnHIKnHgivPQSbLcd/OhH8PGP58SU+j9Hl5MkSZIkST1u6aXhyCNh8mQ44QR46CHYdlv42Mfg2mtz/SYtmkwySZIkSZKkbrfEEnDYYfD003DSSTnptNNOsNVWcNVVJpsWRSaZJEmSJElSjxk2DA45BJ54Av70J3jhBfjMZ2CzzeDSS2H+/HpHqO5ikkmSJEmSJPW4oUNh7Fj43//grLNg5kzYc0/YaCM4/3yYN6/eEaqrTDJJkiRJkqReM2QI7LsvPPIITJiQu8196Uuw/vpwzjkwd26e39wMDQ35dsKEOgetmphkkiRJkiRJvW7QoJxceuABuPji3K1u331h1VXha1+DKVNyAmrKlNwCykRT32eSSZIkSZIk1U1DQ+4299//whVXwBtvwDvvLLjOrFlw9NH1iU+1M8kkSZIkSZLqLiIXBJ87t/LyqVN7Nx51nEkmSZIkSZLUZ4wcWXn+kCHw//5f78aijjHJJEmSJEmS+oxx46CxccF5iy2W5330o7D77vDoo3UJTe0wySRJkiRJkvqMMWNg/Hhoaspd6Jqa4Mwz4dlncwLqhhvgAx+Agw6C556rd7QqFSmlesfQY0aPHp0mTpxY7zAkSZIkSVI3mTEDjjsOTj01t3A6/PA8LbVUvSMbOCLinpTS6PL5tmSSJEmSJEn9xvDh8Ic/wCOPwKc/DT/9Kay9dk46VSsart5hkkmSJEmSJPU7a68Nf/sb3HEHrLcefOMbuRvdpZfCItxpq08zySRJkiRJkvqtLbeEm26CK66AQYNgzz1h663h1lvrHdnAY5JJkiRJkiT1axHwmc/ApElw+ukweTJ85COwxx6ORNebTDJJkiRJkqRFwuDBcMAB8PjjuTj49dfnLnQHHwzPP1/v6BZ9JpkkSZIkSdIiZYkl4Ic/hCefhIMOgjPOyDWcjjkGZs6sd3SLLpNMkiRJkiRpkbTSSnDyyfDww/DJT8Kxx+Zk02mnwV/+As3N0NCQbydMqHe0/d/gegcgSZIkSZLUk0aNgosuyiPRfe97uftcROsodFOmwNix+f6YMfWLs7+zJZMkSZIkSRoQPvQhuOUWGD68NcHUYtYsOPro+sS1qDDJJEmSJEmSBowIeOmlysumTu3dWBY1JpkkSZIkSdKAMnJk5fkpwRe+kEenU8eZZJIkSZIkSQPKuHHQ2LjgvGHDYNdd4cor4f3vzzWapk2rT3z9lUkmSZIkSZI0oIwZA+PHQ1NT7j7X1ASnnw6XXw5PPpkLg599dh6J7vDDq3ev04IilVe6WoSMHj06TZw4sd5hSJIkSZKkfubpp+GYY+Dcc2HJJXOy6bvfhaWWqndk9RcR96SURpfPtyWTJEmSJElSmTXWgHPOgQcegB12gJ/8BNZcE373O3j77XpH1zeZZJIkSZIkSapigw3g0kvhzjth441za6Z11oEzz4R33613dH2LSSZJkiRJkqR2bLEFXH99nlZZBfbfHz7wAbjoIpg/v97R9Q0mmSRJkiRJkmq0/fZwxx25ddOgQbD33rD55nDNNbAIl72uSa8mmSJi54h4LCKeiIgfVFi+W0RMioj7ImJiRHykZNmyEXFxRDwaEY9ExFa9GbskSZIkSRLkEen22AMmTcp1m155BXbeGbbdFm6/vd7R1U+vJZkiYhBwCvBJYH3gixGxftlq/wY2Til9EPgacEbJst8DV6eU1gM2Bh7p8aAlSZIkSZKqGDQI9tkHHn0UTjop3374w7DrrrlgOMCECdDcDA0N+XbChHpG3LN6syXTFsATKaWnUkrvABcAu5WukFKamdJ7jcuWABJARCwNfAz4c7HeOyml13orcEmSJEmSpGqGDoVDDoEnn4Rx4+CWW3KR8K23hq9/HaZMyV3ppkyBsWMX3URTbyaZRgDPlDyeVsxbQETsERGPAv8gt2YCWBOYAZwVEf+NiDMiYomeDliSJEmSJKlWSywBRx0FTz8N3/9+7jo3e/aC68yaBUcfXZ/4elpvJpmiwryFSmKllP5edInbHTiumD0Y2BQ4NaW0CfAWsFBNJ4CIGFvUc5o4Y8aMbglckiRJkiSpVsstB8cfX3351Km9F0tv6s0k0zRg9ZLHqwHTq62cUroFWCsiViy2nZZSurNYfDE56VRpu/EppdEppdHDhw/vnsglSZIkSZI6aOTIjs3v73ozyXQ3MCoi1oiIxYAvAFeUrhARa0dEFPc3BRYDXk4pPQ88ExHrFqtuDzzce6FLkiRJkiR1zLhx0Ni44LzGxjx/UTS4t3aUUno3Ig4BrgEGAWemlB6KiIOK5acBewL7RMRcYDbw+ZJC4IcCE4oE1VPAfr0VuyRJkiRJUkeNGZNvjz46d5EbOTInmFrmL2qiNYez6Bk9enSaOHFivcOQJEmSJElaZETEPSml0eXze7O7nCRJkiRJkhZRJpkkSZIkSZLUZSaZJEmSJEmS1GUmmSRJkiRJktRlJpkkSZIkSZLUZSaZJEmSJEmS1GUmmSRJkiRJktRlJpkkSZIkSZLUZZFSqncMPSYiZgBT6h1HN1gReKneQaguPPYDl8d+4PLYD1we+4HJ4z5weewHLo/9wLUoHfumlNLw8pmLdJJpURERE1NKo+sdh3qfx37g8tgPXB77gctjPzB53Acuj/3A5bEfuAbCsbe7nCRJkiRJkrrMJJMkSZIkSZK6zCRT/zC+3gGobjz2A5fHfuDy2A9cHvuByeM+cHnsBy6P/cC1yB97azJJkiRJkiSpy2zJJEmSJEmSpC4zyVRnEbFzRDwWEU9ExA/aWG/ziJgXEXt1dFv1TV089pMj4oGIuC8iJvZOxOoO7R33iNgmIl4vju19EfHjWrdV39bFY+8134/Vcu0Wx/++iHgoIm7uyLbqu7p47L3u+7EaPvO/V/J5/2Dxv97ytWyrvquLx91rvh+r4dgvExFXRsT9xef9frVu2++klJzqNAGDgCeBNYHFgPuB9ausdwPwT2Cvjmzr1Denrhz7Yv5kYMV6vw6n7j/uwDbAVZ09Z5z65tSVY18s85rvp1ONx35Z4GFgZPF4pVq3deq7U1eOfXHf676fTh29doHPADd0ZlunvjN15bgXj73m++lU4+f9UcAvi/vDgVeKdRe5a96WTPW1BfBESumplNI7wAXAbhXWOxS4BHixE9uqb+rKsVf/1ZXr1mu+f/P4DVy1HPsvAZemlKYCpJRe7MC26ru6cuzVv3X02v0icH4nt1Xf0ZXjrv6tlmOfgKUiIoAlyUmmd2vctl8xyVRfI4BnSh5PK+a9JyJGAHsAp3V0W/VpXTn2kD+kro2IeyJibI9Fqe5W63W7VdGU9l8RsUEHt1Xf1JVjD17z/Vktx34dYLmIuKk4xvt0YFv1XV059uB135/VfO1GRCOwM/lHxQ5tqz6nK8cdvOb7s1qO/cnA+4HpwAPAt1NK82vctl8ZXO8ABrioMK98uL/fAd9PKc3LSc8Obau+qyvHHmDrlNL0iFgJuC4iHk0p3dIDcap71XLc7wWaUkozI2IX4DJgVI3bqu/qyrEHr/n+rJZjPxjYDNgeGAbcHhF31Lit+q5OH/uU0v/wuu/POnLtfga4NaX0Sie2Vd/SleMOXvP9WS3HfifgPmA7YC3yMf5Pjdv2K7Zkqq9pwOolj1cjZzZLjQYuiIjJwF7AHyNi9xq3Vd/VlWNPSml6cfsi8HdyM0v1fe0e95TSGymlmcX9fwJDImLFWrZVn9aVY+8137/Vcu1OA65OKb2VUnoJuAXYuMZt1Xd15dh73fdvHbl2v8CCXaa87vuvrhx3r/n+rZZjvx+5e3RKKT0BPA2sV+O2/YpJpvq6GxgVEWtExGLkD5srSldIKa2RUmpOKTUDFwPfSCldVsu26tM6fewjYomIWAogIpYAdgQe7N3w1UntHveIeF/RV5uI2IL8Of1yLduqT+v0sfea7/dquXYvBz4aEYOLLhRbAo/UuK36rk4fe6/7fq+mazcilgE+Tj4POrSt+qROH3ev+X6vlmM/ldxqlYhYGVgXeKrGbfsVu8vVUUrp3Yg4BLiGXFX+zJTSQxFxULG8Ui2eNrftjbjVdV059sDKwN+L76KDgfNSSlf3dMzquhqP+17AwRHxLjAb+EJKKQFe8/1YV4598Y+I13w/VcuxTyk9EhFXA5OA+cAZKaUHAbzu+6+uHPuIWBOv+36rA//n7QFcm1J6q71te/cVqDO6ctzx//t+rcZjfxxwdkQ8QO4i9/2iBesi97c+8ncXSZIkSZIkqfPsLidJkiRJkqQuM8kkSZIkSZKkLjPJJEmSJEmSpC4zySRJkiRJkqQuM8kkSZIkSZKkLjPJJEmS1IaIaI6IFBGj67DvmyLi5C4+xzZF/Cu2sc5eEeGQw5IkqUtMMkmSpAGrSL60NZ1d7xglSZL6i8H1DkCSJKmOVim5/2ng9LJ5s4HlOvPEETEkpTS3C7FJkiT1K7ZkkiRJA1ZK6fmWCXitfF5K6fWS1Zsi4rqImBURD0fEJ1oWlHRJ2yUi7oqId4CdIjsiIp6MiNkR8UBEfLk0hoj4cURMiYg5EfF8RPylLMyGiPh5RLwUES9GxAkR0VCy/XIRcU5EvFrs4/qI2KCt1x0R+xT7nBURVwErd+4dlCRJamWSSZIkqTbjgD8AGwN3AxdExJJl6/wS+CGwHnAn8DNgf+CbwPrA8cCfIuJTABGxJ3A48A1gFLk11V1lzzkGeBf4MHAI8B3g8yXLzwa2BHYDtgBmAVdHxLBKLyIitiy2GQ98ELgS+Gltb4EkSVJ1kZI1HiVJkiJiL+CilFKUzW8GngYOSin9qZg3ApgGfDSl9P8iYhvgRmCvlNIlxTpLAC8BO6aU/lPyfL8D1kkp7RIR/wccCHygUte6iLgJGJpS2qpk3nXAlJTSARExCvgf8PGU0i3F8mWAqcBhKaUzSmIbnlJ6KSLOK+6XtsQ6A9i//LVLkiR1hC2ZJEmSajOp5P704nalsnUmltxfH1ic3KpoZssEHAysVaxzUbHO0xHx54j4XEQMbWO/Lftu2e/7gfnA7S0Liy5+DxT7r+T9pesXyh9LkiR1mIW/JUmSavNeS6OUUooIWPgHu7dK7rcs+wy5ZdFCz5VSeiYi1gW2B3YAfgP8JCK2TCm9VbpuiVTy3G21PKrWXN3WSpIkqUfYkkmSJKlnPAzMAZpSSk+UTVNaVkopvZ1S+kdK6bvA5sAGwNYd2EcDUNqdbmlgw2JZtW0+VDav/LEkSVKH2ZJJkiSpB6SU3oyIE4ATIjd7ugVYkpzQmZ9SGh8R+5L/H7sTmEku6D0XeLzGfTweEZeTi4mPJY+QNw54AzivymZ/AG6LiCOBi4FtgD068xolSZJK2ZJJkiSp5/wIOIY8gtxDwHXAnuRC4pCTQvsD/wEeLJZ9NqX0dPkTtWE/8oh0VxS3jcDOKaXZlVZOKd1R7PNgcr2nzxYxSpIkdYmjy0mSJEmSJKnLbMkkSZIkSZKkLjPJJEmSJEmSpC4zySRJkiRJkqQuM8kkSZIkSZKkLjPJJEmSJEmSpC4zySRJkiRJkqQuM8kkSZIkSZKkLjPJJEmSJEmSpC4zySRJkiRJkqQu+/8PqQwI1Oe9TQAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 1440x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# PLOT THRESHOLD VS. F1_SCORE\n",
    "plt.figure(figsize=(20,5))\n",
    "plt.plot(thresholds,scores,'-o',color='blue')\n",
    "plt.scatter([best_threshold], [best_score], color='blue', s=300, alpha=1)\n",
    "plt.xlabel('Threshold',size=14)\n",
    "plt.ylabel('Validation F1 Score',size=14)\n",
    "plt.title(f'Threshold vs. F1_Score with Best F1_Score = {best_score:.3f} at Best Threshold = {best_threshold:.3}',size=18)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "1c899af9",
   "metadata": {
    "_cell_guid": "9a805f07-9a58-40d3-87d2-e4a473e34917",
    "_uuid": "ea7fe7d2-867d-4c7b-af7f-504d546928b4",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-03-30T07:53:39.203807Z",
     "iopub.status.busy": "2023-03-30T07:53:39.203063Z",
     "iopub.status.idle": "2023-03-30T07:53:39.295663Z",
     "shell.execute_reply": "2023-03-30T07:53:39.294835Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.200195,
     "end_time": "2023-03-30T07:53:39.298144",
     "exception": false,
     "start_time": "2023-03-30T07:53:39.097949",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "When using optimal threshold...\n",
      "Q0: F1 = 0.3562298953161897\n",
      "Q1: F1 = 0.17127716381872216\n",
      "Q2: F1 = 0.21174605761196247\n",
      "Q3: F1 = 0.4169219764114732\n",
      "Q4: F1 = 0.4644485713344303\n",
      "Q5: F1 = 0.40699890907744285\n",
      "Q6: F1 = 0.43045330026758466\n",
      "Q7: F1 = 0.4666981110886508\n",
      "Q8: F1 = 0.42787401831926464\n",
      "Q9: F1 = 0.4594985319315221\n",
      "Q10: F1 = 0.4597331879279846\n",
      "Q11: F1 = 0.37045291732710084\n",
      "Q12: F1 = 0.47683145445943365\n",
      "Q13: F1 = 0.4821083351608495\n",
      "Q14: F1 = 0.4856576079428281\n",
      "Q15: F1 = 0.4741824794976495\n",
      "Q16: F1 = 0.48324815538816734\n",
      "Q17: F1 = 0.3832259333509157\n",
      "==> Overall F1 = 0.4480892100591143\n"
     ]
    }
   ],
   "source": [
    "print('When using optimal threshold...')\n",
    "for k in range(18):\n",
    "        \n",
    "    # COMPUTE F1 SCORE PER QUESTION\n",
    "    m = f1_score(true[k].values, (oof[k].values>best_threshold).astype('int'), average='macro')\n",
    "    print(f'Q{k}: F1 =',m)\n",
    "    \n",
    "# COMPUTE F1 SCORE OVERALL\n",
    "m = f1_score(true.values.reshape((-1)), (oof.values.reshape((-1))>best_threshold).astype('int'), average='macro')\n",
    "print('==> Overall F1 =',m)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "89018ee5",
   "metadata": {
    "_cell_guid": "95d65e60-09ad-4238-805f-5223f24ef667",
    "_uuid": "477d0cea-8773-4939-be38-da3ed3da405e",
    "papermill": {
     "duration": 0.104631,
     "end_time": "2023-03-30T07:53:39.507418",
     "exception": false,
     "start_time": "2023-03-30T07:53:39.402787",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Infer Test Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "01caa8ec",
   "metadata": {
    "_cell_guid": "8505f0e4-f1a9-444a-b725-595262ab96c1",
    "_uuid": "d1204a07-28c5-497f-9b0d-a9f1ac5ff089",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-03-30T07:53:39.719368Z",
     "iopub.status.busy": "2023-03-30T07:53:39.718548Z",
     "iopub.status.idle": "2023-03-30T07:53:39.782509Z",
     "shell.execute_reply": "2023-03-30T07:53:39.781561Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.174202,
     "end_time": "2023-03-30T07:53:39.785040",
     "exception": false,
     "start_time": "2023-03-30T07:53:39.610838",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "import jo_wilder\n",
    "env = jo_wilder.make_env()\n",
    "iter_test = env.iter_test()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "a858fee6",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-03-30T07:53:39.999782Z",
     "iopub.status.busy": "2023-03-30T07:53:39.999016Z",
     "iopub.status.idle": "2023-03-30T07:53:40.004529Z",
     "shell.execute_reply": "2023-03-30T07:53:40.003742Z"
    },
    "papermill": {
     "duration": 0.115352,
     "end_time": "2023-03-30T07:53:40.006584",
     "exception": false,
     "start_time": "2023-03-30T07:53:39.891232",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "18"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "t"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "576f7689",
   "metadata": {
    "_cell_guid": "de65a33f-8162-4229-8ea7-7ecabbe1fc22",
    "_uuid": "daf327d3-91f7-480b-a2b2-faf9f5d6f754",
    "execution": {
     "iopub.execute_input": "2023-03-30T07:53:40.216881Z",
     "iopub.status.busy": "2023-03-30T07:53:40.215959Z",
     "iopub.status.idle": "2023-03-30T07:53:40.661745Z",
     "shell.execute_reply": "2023-03-30T07:53:40.660641Z"
    },
    "papermill": {
     "duration": 0.554233,
     "end_time": "2023-03-30T07:53:40.665205",
     "exception": false,
     "start_time": "2023-03-30T07:53:40.110972",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "This version of the API is not optimized and should not be used to estimate the runtime of your code on the hidden test set.\n",
      "0\n",
      "1\n",
      "2\n"
     ]
    }
   ],
   "source": [
    "tmp=[]\n",
    "tests=[]\n",
    "sample_submissions=[]\n",
    "i=0\n",
    "for (test, sample_submission) in iter_test:\n",
    "    tests.append(test)\n",
    "#     sample_submissions.append(sample_submission)\n",
    "    print(i)\n",
    "    i+=1\n",
    "    x = feature_engineer(test)\n",
    "    y = sample_submission.copy()\n",
    "    y['session'] = y.session_id.apply(lambda x: int(x.split('_')[0]) )\n",
    "    y['q'] = y.session_id.apply(lambda x: int(x.split('_')[-1][1:]) )\n",
    "    y=y.set_index('session_id')\n",
    "    for t in y['q'].unique():\n",
    "        if t<=3: grp = '0-4'\n",
    "        elif t<=13: grp = '5-12'\n",
    "        elif t<=22: grp = '13-22'\n",
    "        x_ = x.loc[x.level_group == grp]\n",
    "        users = x_.index.values\n",
    "        clf = models[f'{grp}_{t}']\n",
    "        p = clf.predict_proba(x_[FEATURES].astype('float32'))[:,1]\n",
    "        y.loc[x_.index.map(lambda x:str(x)+f'_q{t}'),'correct']=p>best_threshold\n",
    "    y.correct=y.correct.astype('int')\n",
    "    sample_submission['correct']=y.loc[sample_submission.session_id,'correct'].values\n",
    "    tmp.append(sample_submission)\n",
    "    env.predict(sample_submission)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "6df009aa",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-03-30T07:53:40.876582Z",
     "iopub.status.busy": "2023-03-30T07:53:40.875972Z",
     "iopub.status.idle": "2023-03-30T07:53:40.890711Z",
     "shell.execute_reply": "2023-03-30T07:53:40.889580Z"
    },
    "papermill": {
     "duration": 0.124048,
     "end_time": "2023-03-30T07:53:40.894678",
     "exception": false,
     "start_time": "2023-03-30T07:53:40.770630",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>session_id</th>\n",
       "      <th>correct</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>20090109393214576_q1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>20090312143683264_q1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>20090312331414616_q1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>20090109393214576_q2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>20090312143683264_q2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>20090312331414616_q2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>20090109393214576_q3</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>20090312143683264_q3</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>20090312331414616_q3</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>20090109393214576_q4</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>20090312143683264_q4</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>20090312331414616_q4</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>20090109393214576_q5</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>20090312143683264_q5</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>20090312331414616_q5</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>20090109393214576_q6</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>20090312143683264_q6</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>20090312331414616_q6</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>20090109393214576_q7</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>20090312143683264_q7</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20</th>\n",
       "      <td>20090312331414616_q7</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21</th>\n",
       "      <td>20090109393214576_q8</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22</th>\n",
       "      <td>20090312143683264_q8</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23</th>\n",
       "      <td>20090312331414616_q8</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24</th>\n",
       "      <td>20090109393214576_q9</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25</th>\n",
       "      <td>20090312143683264_q9</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>26</th>\n",
       "      <td>20090312331414616_q9</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>27</th>\n",
       "      <td>20090109393214576_q10</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>28</th>\n",
       "      <td>20090312143683264_q10</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>29</th>\n",
       "      <td>20090312331414616_q10</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>30</th>\n",
       "      <td>20090109393214576_q11</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>31</th>\n",
       "      <td>20090312143683264_q11</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>32</th>\n",
       "      <td>20090312331414616_q11</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>33</th>\n",
       "      <td>20090109393214576_q12</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>34</th>\n",
       "      <td>20090312143683264_q12</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>35</th>\n",
       "      <td>20090312331414616_q12</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>36</th>\n",
       "      <td>20090109393214576_q13</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>37</th>\n",
       "      <td>20090312143683264_q13</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>38</th>\n",
       "      <td>20090312331414616_q13</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>39</th>\n",
       "      <td>20090109393214576_q14</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>40</th>\n",
       "      <td>20090312143683264_q14</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>41</th>\n",
       "      <td>20090312331414616_q14</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>42</th>\n",
       "      <td>20090109393214576_q15</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>43</th>\n",
       "      <td>20090312143683264_q15</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>44</th>\n",
       "      <td>20090312331414616_q15</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>45</th>\n",
       "      <td>20090109393214576_q16</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>46</th>\n",
       "      <td>20090312143683264_q16</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>47</th>\n",
       "      <td>20090312331414616_q16</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>48</th>\n",
       "      <td>20090109393214576_q17</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>49</th>\n",
       "      <td>20090312143683264_q17</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50</th>\n",
       "      <td>20090312331414616_q17</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>51</th>\n",
       "      <td>20090109393214576_q18</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>52</th>\n",
       "      <td>20090312143683264_q18</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>53</th>\n",
       "      <td>20090312331414616_q18</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               session_id  correct\n",
       "0    20090109393214576_q1        1\n",
       "1    20090312143683264_q1        1\n",
       "2    20090312331414616_q1        1\n",
       "3    20090109393214576_q2        1\n",
       "4    20090312143683264_q2        1\n",
       "5    20090312331414616_q2        1\n",
       "6    20090109393214576_q3        1\n",
       "7    20090312143683264_q3        1\n",
       "8    20090312331414616_q3        1\n",
       "9    20090109393214576_q4        1\n",
       "10   20090312143683264_q4        1\n",
       "11   20090312331414616_q4        1\n",
       "12   20090109393214576_q5        1\n",
       "13   20090312143683264_q5        1\n",
       "14   20090312331414616_q5        1\n",
       "15   20090109393214576_q6        1\n",
       "16   20090312143683264_q6        1\n",
       "17   20090312331414616_q6        1\n",
       "18   20090109393214576_q7        1\n",
       "19   20090312143683264_q7        1\n",
       "20   20090312331414616_q7        1\n",
       "21   20090109393214576_q8        1\n",
       "22   20090312143683264_q8        1\n",
       "23   20090312331414616_q8        1\n",
       "24   20090109393214576_q9        1\n",
       "25   20090312143683264_q9        1\n",
       "26   20090312331414616_q9        1\n",
       "27  20090109393214576_q10        1\n",
       "28  20090312143683264_q10        1\n",
       "29  20090312331414616_q10        1\n",
       "30  20090109393214576_q11        1\n",
       "31  20090312143683264_q11        1\n",
       "32  20090312331414616_q11        1\n",
       "33  20090109393214576_q12        1\n",
       "34  20090312143683264_q12        1\n",
       "35  20090312331414616_q12        1\n",
       "36  20090109393214576_q13        0\n",
       "37  20090312143683264_q13        0\n",
       "38  20090312331414616_q13        0\n",
       "39  20090109393214576_q14        1\n",
       "40  20090312143683264_q14        1\n",
       "41  20090312331414616_q14        1\n",
       "42  20090109393214576_q15        1\n",
       "43  20090312143683264_q15        1\n",
       "44  20090312331414616_q15        1\n",
       "45  20090109393214576_q16        1\n",
       "46  20090312143683264_q16        1\n",
       "47  20090312331414616_q16        1\n",
       "48  20090109393214576_q17        1\n",
       "49  20090312143683264_q17        1\n",
       "50  20090312331414616_q17        1\n",
       "51  20090109393214576_q18        1\n",
       "52  20090312143683264_q18        1\n",
       "53  20090312331414616_q18        1"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tmp=pd.concat(tmp)\n",
    "tmp.reset_index(drop=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8a4622e6",
   "metadata": {
    "_cell_guid": "d64c352f-8b1e-4a8d-a829-628ea4a267dc",
    "_uuid": "4fd61cdb-b2e3-4ff9-9516-a5fd4ef0496b",
    "papermill": {
     "duration": 0.104238,
     "end_time": "2023-03-30T07:53:41.103276",
     "exception": false,
     "start_time": "2023-03-30T07:53:40.999038",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# EDA submission.csv"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "8dbe4052",
   "metadata": {
    "_cell_guid": "730145d2-6d3b-4144-ba93-90f251f9492d",
    "_uuid": "0a38a58d-11c4-42ae-aa8b-c481e20d338f",
    "execution": {
     "iopub.execute_input": "2023-03-30T07:53:41.313506Z",
     "iopub.status.busy": "2023-03-30T07:53:41.313011Z",
     "iopub.status.idle": "2023-03-30T07:53:41.330242Z",
     "shell.execute_reply": "2023-03-30T07:53:41.329242Z"
    },
    "papermill": {
     "duration": 0.125059,
     "end_time": "2023-03-30T07:53:41.332564",
     "exception": false,
     "start_time": "2023-03-30T07:53:41.207505",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(54, 2)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>session_id</th>\n",
       "      <th>correct</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>20090109393214576_q1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>20090312143683264_q1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>20090312331414616_q1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>20090109393214576_q2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>20090312143683264_q2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>20090312331414616_q2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>20090109393214576_q3</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>20090312143683264_q3</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>20090312331414616_q3</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>20090109393214576_q4</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>20090312143683264_q4</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>20090312331414616_q4</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>20090109393214576_q5</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>20090312143683264_q5</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>20090312331414616_q5</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>20090109393214576_q6</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>20090312143683264_q6</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>20090312331414616_q6</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>20090109393214576_q7</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>20090312143683264_q7</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20</th>\n",
       "      <td>20090312331414616_q7</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21</th>\n",
       "      <td>20090109393214576_q8</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22</th>\n",
       "      <td>20090312143683264_q8</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23</th>\n",
       "      <td>20090312331414616_q8</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24</th>\n",
       "      <td>20090109393214576_q9</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25</th>\n",
       "      <td>20090312143683264_q9</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>26</th>\n",
       "      <td>20090312331414616_q9</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>27</th>\n",
       "      <td>20090109393214576_q10</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>28</th>\n",
       "      <td>20090312143683264_q10</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>29</th>\n",
       "      <td>20090312331414616_q10</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>30</th>\n",
       "      <td>20090109393214576_q11</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>31</th>\n",
       "      <td>20090312143683264_q11</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>32</th>\n",
       "      <td>20090312331414616_q11</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>33</th>\n",
       "      <td>20090109393214576_q12</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>34</th>\n",
       "      <td>20090312143683264_q12</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>35</th>\n",
       "      <td>20090312331414616_q12</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>36</th>\n",
       "      <td>20090109393214576_q13</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>37</th>\n",
       "      <td>20090312143683264_q13</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>38</th>\n",
       "      <td>20090312331414616_q13</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>39</th>\n",
       "      <td>20090109393214576_q14</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>40</th>\n",
       "      <td>20090312143683264_q14</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>41</th>\n",
       "      <td>20090312331414616_q14</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>42</th>\n",
       "      <td>20090109393214576_q15</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>43</th>\n",
       "      <td>20090312143683264_q15</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>44</th>\n",
       "      <td>20090312331414616_q15</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>45</th>\n",
       "      <td>20090109393214576_q16</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>46</th>\n",
       "      <td>20090312143683264_q16</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>47</th>\n",
       "      <td>20090312331414616_q16</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>48</th>\n",
       "      <td>20090109393214576_q17</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>49</th>\n",
       "      <td>20090312143683264_q17</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50</th>\n",
       "      <td>20090312331414616_q17</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>51</th>\n",
       "      <td>20090109393214576_q18</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>52</th>\n",
       "      <td>20090312143683264_q18</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>53</th>\n",
       "      <td>20090312331414616_q18</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               session_id  correct\n",
       "0    20090109393214576_q1        1\n",
       "1    20090312143683264_q1        1\n",
       "2    20090312331414616_q1        1\n",
       "3    20090109393214576_q2        1\n",
       "4    20090312143683264_q2        1\n",
       "5    20090312331414616_q2        1\n",
       "6    20090109393214576_q3        1\n",
       "7    20090312143683264_q3        1\n",
       "8    20090312331414616_q3        1\n",
       "9    20090109393214576_q4        1\n",
       "10   20090312143683264_q4        1\n",
       "11   20090312331414616_q4        1\n",
       "12   20090109393214576_q5        1\n",
       "13   20090312143683264_q5        1\n",
       "14   20090312331414616_q5        1\n",
       "15   20090109393214576_q6        1\n",
       "16   20090312143683264_q6        1\n",
       "17   20090312331414616_q6        1\n",
       "18   20090109393214576_q7        1\n",
       "19   20090312143683264_q7        1\n",
       "20   20090312331414616_q7        1\n",
       "21   20090109393214576_q8        1\n",
       "22   20090312143683264_q8        1\n",
       "23   20090312331414616_q8        1\n",
       "24   20090109393214576_q9        1\n",
       "25   20090312143683264_q9        1\n",
       "26   20090312331414616_q9        1\n",
       "27  20090109393214576_q10        1\n",
       "28  20090312143683264_q10        1\n",
       "29  20090312331414616_q10        1\n",
       "30  20090109393214576_q11        1\n",
       "31  20090312143683264_q11        1\n",
       "32  20090312331414616_q11        1\n",
       "33  20090109393214576_q12        1\n",
       "34  20090312143683264_q12        1\n",
       "35  20090312331414616_q12        1\n",
       "36  20090109393214576_q13        0\n",
       "37  20090312143683264_q13        0\n",
       "38  20090312331414616_q13        0\n",
       "39  20090109393214576_q14        1\n",
       "40  20090312143683264_q14        1\n",
       "41  20090312331414616_q14        1\n",
       "42  20090109393214576_q15        1\n",
       "43  20090312143683264_q15        1\n",
       "44  20090312331414616_q15        1\n",
       "45  20090109393214576_q16        1\n",
       "46  20090312143683264_q16        1\n",
       "47  20090312331414616_q16        1\n",
       "48  20090109393214576_q17        1\n",
       "49  20090312143683264_q17        1\n",
       "50  20090312331414616_q17        1\n",
       "51  20090109393214576_q18        1\n",
       "52  20090312143683264_q18        1\n",
       "53  20090312331414616_q18        1"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv('submission.csv')\n",
    "print( df.shape )\n",
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "e1b51ef1",
   "metadata": {
    "_cell_guid": "399db6cb-cf89-402f-8db4-a23fd335e23d",
    "_uuid": "6e687bd9-814c-4c90-ac93-a46bf8b41b9c",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-03-30T07:53:41.603571Z",
     "iopub.status.busy": "2023-03-30T07:53:41.603000Z",
     "iopub.status.idle": "2023-03-30T07:53:41.609533Z",
     "shell.execute_reply": "2023-03-30T07:53:41.608628Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.171944,
     "end_time": "2023-03-30T07:53:41.612255",
     "exception": false,
     "start_time": "2023-03-30T07:53:41.440311",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.9444444444444444\n"
     ]
    }
   ],
   "source": [
    "print(df.correct.mean())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fd7dfa66",
   "metadata": {
    "papermill": {
     "duration": 0.104783,
     "end_time": "2023-03-30T07:53:41.822764",
     "exception": false,
     "start_time": "2023-03-30T07:53:41.717981",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.12"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 207.74182,
   "end_time": "2023-03-30T07:53:42.956193",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2023-03-30T07:50:15.214373",
   "version": "2.3.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
