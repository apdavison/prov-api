
EXAMPLES = {
    "DataAnalysis": {
        "end_time": "2021-05-28T16:32:58.597000+00:00",
        "environment": {
            "name": "SpiNNaker default 2021-10-13",
            "hardware": "SpiNNaker",
            "configuration": {
                "parameter1": "value1",
                "parameter2": "value2"
            },
            "software": [
                {
                    "software_name": "numpy",
                    "software_version": "1.19.3"
                },
                {
                    "software_name": "neo",
                    "software_version": "0.9.0"
                },
                {
                    "software_name": "spyNNaker",
                    "software_version": "5.0.0"
                }
            ],
            "description": "Default environment on SpiNNaker 1M core machine as of 2020-10-13 (not really, this is just for example purposes)."
        },
        "id": "00000000-0000-0000-0000-000000000000",
        "input": [
            {
                "description": "Demonstration data for validation framework",
                "format": "application/json",
                "hash": {
                    "algorithm": "SHA-1",
                    "value": "716c29320b1e329196ce15d904f7d4e3c7c46685"
                },
                "location": "https://object.cscs.ch/v1/AUTH_c0a333ecf7c045809321ce9d9ecdfdea/VF_paper_demo/obs_data/InputResistance_data.json",
                "file_name": "InputResistance_data.json",
                "size": 34
            },
            {
                "software_name": "Elephant",
                "software_version": "0.10.0"
            }
        ],
        "launch_config": {
            "executable": "/usr/bin/python",
            "arguments": [
                "-Werror"
            ],
            "environment_variables": {
                "items": [
                    {
                        "name": "COLLAB_ID",
                        "value": "myspace"
                    }
                ]
            }
        },
        "output": [
            {
                "description": "Demonstration data for validation framework",
                "format": "application/json",
                "hash": {
                    "algorithm": "SHA-1",
                    "value": "716c29320b1e329196ce15d904f7d4e3c7c46685"
                },
                "location": "https://object.cscs.ch/v1/AUTH_c0a333ecf7c045809321ce9d9ecdfdea/VF_paper_demo/obs_data/InputResistance_data.json",
                "file_name": "InputResistance_data.json",
                "size": 34
            }
        ],
        "project_id": "fake-space",
        "resource_usage": [
            {
                "value": 1017.3,
                "units": "core-hour"
            }
        ],
        "start_time": "2021-05-28T16:32:58.597000+00:00",
        "started_by": {
            "family_name": "Destexhe",
            "given_name": "Alain",
            "orcid": "https://orcid.org/0000-0001-7405-0455"
        },
        "status": "queued",
        "tags": [
            "string"
        ],
        "type": "data analysis"
    }
}
