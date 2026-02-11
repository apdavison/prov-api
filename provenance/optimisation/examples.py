
EXAMPLES = {
    "Optimisation": {
        "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
        "type": "optimization",
        "input": [
            {
                "model_version_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"
            },
            {
                "software_name": "NEST",
                "software_version": "3.1.0"
            }
        ],
        "output": [
            {
                "model_version_id": "4fa85f64-5717-4562-b3fc-2c963f66afa7"
            },
            {
                "model_version_id": "5fa85f64-5717-4562-b3fc-2c963f66afa8"
            },
            {
                "model_version_id": "6fa85f64-5717-4562-b3fc-2c963f66afa9"
            },

        ],
        "environment": {
            "name": "Piz Daint 2022",
            "hardware": "Piz Daint",
            "configuration": {
                "items": [
                    {
                        "name": "parameter1",
                        "value": "value1"
                    },
                    {
                        "name": "parameter2",
                        "value": "value2"
                    }
                ],
                "description": "hardware configuration for Piz Daint in 2022"
            },
            "software": [
                {
                    "software_name": "numpy",
                    "software_version": "1.19.3"
                },
                {
                    "software_name": "BluePyOpt",
                    "software_version": "0.9.0"
                }
            ],
            "description": "Environment for project ich009 on Piz Daint"
        },
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
        "start_time": "2022-05-28T16:32:58.597Z",
        "end_time": "2022-05-28T16:32:58.597Z",
        "started_by": {
            "family_name": "Davison",
            "given_name": "Andrew"
        },
        "status": "completed",
        "resource_usage": [
            {
                "value": 123017.3,
                "units": "core-hour"
            }
        ],
        "tags": [
            "string"
        ]
    }
}
